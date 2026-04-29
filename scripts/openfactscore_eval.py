"""
OpenFActScore-style evaluation (source faithfulness variant) — fully open backend.

Implements the same two-stage pipeline as factscore_eval.py (Min et al. 2023)
but replaces the closed-source GPT-4o-mini judge with two HuggingFace models,
following Lage & Ostermann (2025) — OpenFActScore. Models are aligned with the
official repo (https://github.com/lflage/OpenFActScore):
  - AFG (Atomic Fact Generation): allenai/OLMo-2-1124-7B-SFT
  - AFV (Atomic Fact Validation): google/gemma-3-4b-it

Verification source = E_0 of the chain (source faithfulness, not Wikipedia).
This matches the existing factscore_eval.py setup and is the variant relevant
to RQ3 (factuality preservation across iterative rewriting).

Output schema mirrors factscore_eval.py so downstream visualization scripts
work without changes:
  qid, group, instruction_type, run, step, instruction_used,
  n_facts, n_supported, n_not_supported, n_contradicted,
  init_score, factscore

Plus a sidecar `_details.csv` with one row per (chain, step, fact).
"""

import argparse
import json
import os
import re
import string
import time
from pathlib import Path

import nltk
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "results" / "rewriting_chains_15q.csv"
DEFAULT_DEMOS = Path(os.path.expanduser("~/.cache/factscore/demos/demons.json"))

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

AFG_MODEL_ID = "allenai/OLMo-2-1124-7B-SFT"
AFV_MODEL_ID = "google/gemma-3-4b-it"

N_FIXED_DEMOS = 7
K_BM25 = 1
AFG_MAX_NEW_TOKENS = 256
AFV_MAX_NEW_TOKENS = 8

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Sentence splitting (same as factscore_eval.py)
# ---------------------------------------------------------------------------

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    return [m.group() for m in re.finditer(pattern, text)]


def fix_sentence_splitter(sentences, initials):
    for initial in initials:
        if not any(initial in s for s in sentences):
            alpha1, alpha2 = [s.strip() for s in initial.split(".") if s.strip()]
            for i, sent in enumerate(sentences):
                if sent.endswith(alpha1 + "."):
                    if i + 1 < len(sentences) and sentences[i + 1].startswith(alpha2 + "."):
                        sentences[i] = sent + " " + sentences[i + 1]
                        sentences = sentences[:i + 1] + sentences[i + 2:]
                        break
    merged = []
    for sent in sentences:
        if merged and len(sent.split()) <= 1:
            merged[-1] = merged[-1] + " " + sent
        else:
            merged.append(sent)
    return merged


def sentences_from_text(text):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences = []
    for para in paragraphs:
        initials = detect_initials(para)
        curr = sent_tokenize(para)
        curr = fix_sentence_splitter(curr, initials)
        sentences.extend(curr)
    return sentences


# ---------------------------------------------------------------------------
# Atomic Fact Generation (AFG) — OLMo-2 7B Instruct
# ---------------------------------------------------------------------------

# System prompt from OpenFActScore (Lage & Ostermann 2025), appendix A.1.
# We embed the BM25-selected demo into the system prompt as the paper does.
AFG_SYSTEM_PROMPT_TEMPLATE = """You are an annotator that breaks down sentences into independent facts, short statements that each contain one piece of information contained in the given sentence. In the next paragraphs you have examples of sentences broken down in atomic facts. You have to complete the example given by the user. Do not add new entities, do not deviate from the subject of the sentence given by the user, do not hallucinate, do not repeat facts in the system prompt. List the sentences using "- " as bullet.

{demos}"""

AFG_USER_PROMPT_TEMPLATE = "Please break down the following sentence into independent facts: {sentence}"


def build_afg_demos_block(demons, demon_keys, bm25, sentence):
    """Construct the demonstration block (fixed demos + 1 BM25-retrieved demo)."""
    parts = []
    for i in range(N_FIXED_DEMOS):
        key = demon_keys[i]
        parts.append(f"Please break down the following sentence into independent facts: {key}")
        for fact in demons[key]:
            parts.append(f"- {fact}")
        parts.append("")

    tokenized_query = sentence.split(" ")
    top_matches = bm25.get_top_n(tokenized_query, demon_keys, K_BM25)
    for match in top_matches:
        parts.append(f"Please break down the following sentence into independent facts: {match}")
        for fact in demons[match]:
            parts.append(f"- {fact}")
        parts.append("")

    return "\n".join(parts)


def parse_atomic_facts(generated_text):
    """Pull out '- fact' lines until the model stops emitting bullets."""
    facts = []
    for line in generated_text.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            fact = line[2:].strip()
            if fact and fact[-1] not in string.punctuation:
                fact = fact + "."
            if fact:
                facts.append(fact)
        elif not line:
            if facts:
                break
        else:
            # non-bullet, non-empty: stop here (model has moved on to new content)
            if facts:
                break
    return facts


# ---------------------------------------------------------------------------
# Atomic Fact Validation (AFV) — Gemma 3 12B IT
# ---------------------------------------------------------------------------

# Adapted from the original FActScore prompt (Min et al. 2023, §2.3) and
# OpenFActScore appendix A.2, but the "knowledge source" is replaced by the
# chain's E_0 — that's the source-faithfulness variant.
AFV_SYSTEM_PROMPT = (
    "You are an annotator that verifies the factuality of a sentence according "
    "to a given source text. You answer only True or False and provide no "
    "further explanations."
)

AFV_USER_PROMPT_TEMPLATE = """Answer the question about {topic} based on the given context.

[Title: {topic}]
[Text: {source}]

Input: {claim} True or False?
Answer:"""


def parse_afv_label(generated_text):
    """Look for the first True/False token in the model output."""
    text = generated_text.strip()
    # match whole word, case-insensitive
    m = re.search(r"\b(true|false)\b", text, re.IGNORECASE)
    if not m:
        return "NOT_SUPPORTED"
    return "SUPPORTED" if m.group(1).lower() == "true" else "NOT_SUPPORTED"


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class HFChatModel:
    """Wraps an HF causal LM with chat-template + system-prompt support."""

    def __init__(self, model_id, role_label):
        print(f"[{role_label}] loading {model_id} ...", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[{role_label}] loaded in {time.time()-t0:.1f}s · device map: {getattr(self.model, 'hf_device_map', 'n/a')}", flush=True)

    @torch.no_grad()
    def generate(self, system_prompt, user_prompt, max_new_tokens):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # fallback: simple concatenation
            sys_block = f"{system_prompt}\n\n" if system_prompt else ""
            text = f"{sys_block}{user_prompt}"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def extract_atomic_facts(afg_model, text, demons, demon_keys, bm25):
    sentences = sentences_from_text(text)
    all_facts = []
    for sent in sentences:
        if sent.lower().startswith(("sure", "here are", "please", "i hope")):
            continue
        demos_block = build_afg_demos_block(demons, demon_keys, bm25, sent)
        system_prompt = AFG_SYSTEM_PROMPT_TEMPLATE.format(demos=demos_block)
        user_prompt = AFG_USER_PROMPT_TEMPLATE.format(sentence=sent)
        out = afg_model.generate(system_prompt, user_prompt, AFG_MAX_NEW_TOKENS)
        all_facts.extend(parse_atomic_facts(out))
    return all_facts


def validate_facts(afv_model, source, facts, topic):
    results = []
    for fact in facts:
        user_prompt = AFV_USER_PROMPT_TEMPLATE.format(
            topic=topic, source=source.strip(), claim=fact,
        )
        out = afv_model.generate(AFV_SYSTEM_PROMPT, user_prompt, AFV_MAX_NEW_TOKENS)
        label = parse_afv_label(out)
        results.append({"fact": fact, "label": label, "raw": out})
    return results


def compute_factscore(afg_model, afv_model, source, generated, topic, demons, demon_keys, bm25, gamma=10):
    facts = extract_atomic_facts(afg_model, generated, demons, demon_keys, bm25)
    if not facts:
        return {
            "n_facts": 0,
            "n_supported": 0,
            "n_not_supported": 0,
            "n_contradicted": 0,
            "init_score": None,
            "factscore": None,
            "verified_facts": [],
        }

    verified = validate_facts(afv_model, source, facts, topic)
    counts = {"SUPPORTED": 0, "NOT_SUPPORTED": 0, "CONTRADICTED": 0}
    # Note: OpenFActScore uses binary True/False → no CONTRADICTED label
    # We keep the column for output-schema compatibility with factscore_eval.py
    # but it will always be 0 for this backend.
    for v in verified:
        counts[v["label"]] = counts.get(v["label"], 0) + 1

    init_score = counts["SUPPORTED"] / len(facts)
    length_penalty = min(1.0, len(facts) / gamma) if gamma > 0 else 1.0
    final_score = init_score * length_penalty

    return {
        "n_facts": len(facts),
        "n_supported": counts["SUPPORTED"],
        "n_not_supported": counts["NOT_SUPPORTED"],
        "n_contradicted": counts.get("CONTRADICTED", 0),
        "init_score": init_score,
        "factscore": final_score,
        "verified_facts": verified,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenFActScore (OLMo + Gemma) over a chain CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV, help=f"Chain CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMOS, help=f"Demons JSON for AFG few-shot (default: {DEFAULT_DEMOS})")
    parser.add_argument("--afg-model", default=AFG_MODEL_ID, help=f"HF model id for AFG (default: {AFG_MODEL_ID})")
    parser.add_argument("--afv-model", default=AFV_MODEL_ID, help=f"HF model id for AFV (default: {AFV_MODEL_ID})")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test: only score the first N (step>0) rows.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Chain CSV not found: {args.input}")
    if not args.demos.exists():
        raise FileNotFoundError(
            f"demons.json not found: {args.demos}\n"
            "Download it from the original FActScore repo and place it at the path above."
        )

    print("=" * 70)
    print("OpenFActScore — source-faithfulness against E_0 (chain step 0)")
    print(f"  AFG: {args.afg_model}")
    print(f"  AFV: {args.afv_model}")
    print("=" * 70)

    with open(args.demos) as f:
        demons = json.load(f)
    demon_keys = list(demons.keys())
    tokenized_corpus = [doc.split(" ") for doc in demon_keys]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"\nLoaded {len(demon_keys)} AFG demonstrations")

    df = pd.read_csv(args.input)
    df = df.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    sources = df[df["step"] == 0].set_index(CHAIN_KEYS)["text"].to_dict()
    if not sources:
        raise RuntimeError("No step=0 rows found in the input CSV.")
    print(f"Loaded {len(df)} rows · {len(sources)} chains (step=0 sources)")

    to_eval = df[df["step"] > 0].copy()
    if args.limit:
        to_eval = to_eval.head(args.limit)
        print(f"*** SMOKE TEST: limiting to first {args.limit} rows ***")

    out_scores_path = args.input.with_name(args.input.stem + "_openfactscore.csv")
    out_details_path = args.input.with_name(args.input.stem + "_openfactscore_details.csv")
    print(f"Output scores:  {out_scores_path}")
    print(f"Output details: {out_details_path}")

    # Resume support: skip rows already in the output CSV
    done_keys = set()
    if out_scores_path.exists():
        prev = pd.read_csv(out_scores_path)
        done_keys = {tuple(row[k] for k in CHAIN_KEYS + ["step"]) for _, row in prev.iterrows()}
        print(f"Resume: {len(done_keys)} (chain, step) rows already scored — will skip them.")

    afg = HFChatModel(args.afg_model, "AFG")
    afv = HFChatModel(args.afv_model, "AFV")

    total = len(to_eval)
    t_start = time.time()
    n_done = 0

    for i, (_, row) in enumerate(to_eval.iterrows(), start=1):
        chain_id = tuple(row[k] for k in CHAIN_KEYS)
        key = chain_id + (int(row["step"]),)
        if key in done_keys:
            continue

        source = sources.get(chain_id)
        if source is None:
            continue

        label = f"{row['group']}/{row['instruction_type']}/run{row['run']}/step{row['step']}"
        t0 = time.time()
        print(f"[{i}/{total}] {label} ...", end=" ", flush=True)

        result = compute_factscore(
            afg, afv, source, row["text"], row["qid"], demons, demon_keys, bm25,
        )
        elapsed = time.time() - t0

        score_row = pd.DataFrame([{
            **{k: row[k] for k in CHAIN_KEYS},
            "step": int(row["step"]),
            "instruction_used": row.get("instruction_used"),
            **{k: v for k, v in result.items() if k != "verified_facts"},
        }])
        write_header = not out_scores_path.exists()
        score_row.to_csv(out_scores_path, mode="a", header=write_header, index=False, encoding="utf-8")

        if result["verified_facts"]:
            details_rows = pd.DataFrame([
                {
                    **{k: row[k] for k in CHAIN_KEYS},
                    "step": int(row["step"]),
                    "fact": vf["fact"],
                    "label": vf["label"],
                    "raw": vf["raw"],
                }
                for vf in result["verified_facts"]
            ])
            write_header = not out_details_path.exists()
            details_rows.to_csv(out_details_path, mode="a", header=write_header, index=False, encoding="utf-8")

        n_done += 1
        if result["factscore"] is None:
            print(f"no facts  [{elapsed:.1f}s]")
        else:
            print(
                f"facts={result['n_facts']:>2}  supp={result['n_supported']:>2}  "
                f"init={result['init_score']:.3f}  score={result['factscore']:.3f}  "
                f"[{elapsed:.1f}s]",
                flush=True,
            )

        avg = (time.time() - t_start) / max(n_done, 1)
        remaining = (total - i) * avg
        if i % 10 == 0:
            print(f"   ETA: {remaining/60:.1f} min  (avg {avg:.1f}s/row)", flush=True)

    print(f"\nTotal: {time.time() - t_start:.1f}s  ({(time.time() - t_start)/60:.1f} min)")
    print(f"Saved: {out_scores_path}")
    print(f"Saved: {out_details_path}")

    print("\n" + "=" * 70)
    print("OpenFActScore — median per (instruction_type, step)")
    print("=" * 70)
    out = pd.read_csv(out_scores_path)
    pivot = out.pivot_table(
        index="instruction_type", columns="step", values="init_score", aggfunc="median",
    )
    print(pivot.round(3))


if __name__ == "__main__":
    main()
