"""
OpenFActScore (source-faithfulness) for FictionalQA rewriting chains.

Identical pipeline to the MuSiQue / NewsQA versions, adapted for the
FictionalQA CSV schema.

Key differences
---------------
- The chains CSV already carries question + gold_answer + aliases (and the
  fiction_id, style, event_id metadata).
- The AFV "topic" string is derived from the first non-empty line of E_0
  (typically the headline/title of the fictional document).
- These documents are *fictional* — there is no risk that the AFV model will
  validate facts based on prior knowledge of real-world events. The AFV
  judgment must come purely from the source text in front of it.

Two stages
----------
  AFG: allenai/OLMo-2-1124-7B-SFT  — Atomic Fact Generation
  AFV: google/gemma-3-4b-it         — Atomic Fact Validation

Source for AFV is E_0 (the step=0 fiction text for each chain).

Output
------
  rewriting_chains_fictionalqa_openfactscore.csv    — one row per (chain, step)
  rewriting_chains_fictionalqa_openfactscore_details.csv — one row per (chain, step, fact)
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = REPO_ROOT / "results" / "fictionalqa" / "rewriting_chains_fictionalqa.csv"
DEFAULT_DEMOS = REPO_ROOT / "data" / "demons.json"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

AFG_MODEL_ID = "allenai/OLMo-2-1124-7B-SFT"
AFV_MODEL_ID = "google/gemma-3-4b-it"

K_BM25 = 1
AFG_MAX_NEW_TOKENS = 256
AFV_MAX_NEW_TOKENS = 8

# Max chars of the first line of the article used as the AFV topic. NewsQA
# articles often start with a CNN dateline ("(CNN) -- ...") so we keep enough
# of it to convey what the article is about, but avoid stuffing the whole lead.
TOPIC_MAX_CHARS = 200


def derive_topic(text: str, fallback: str) -> str:
    """Pick a short, semantically meaningful topic string from the document.

    FictionalQA fiction_ids look like "event_000_style_blog_num_000" — useless
    as topics. Take the first non-empty line of E_0 (typically the document
    title / headline / opening sentence) and trim it.
    """
    if not isinstance(text, str):
        return fallback
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:TOPIC_MAX_CHARS]
    return fallback

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Sentence splitting
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
# AFG
# ---------------------------------------------------------------------------

AFG_SYSTEM_INSTRUCT = (
    "\n                You are an annotator that breaks down sentences into "
    "independent facts, short statements that each contain one piece of "
    "information contained in the given sentence.\n"
    "                in the next paragraphs you have examples of sentences "
    "broken down in atomic facts. \n"
    "                You have to complete the example given by the user.\n"
    "                Do not add new entities, do not deviate from the subject "
    "of the sentence given by the user, do not hallucinate, do not repeat "
    "facts in the system prompt.\n"
    "                List the sentences using -\n                "
)


def build_afg_demos_block(demons, demon_keys, bm25, sentence):
    tokenized_query = sentence.split(" ")
    top_matches = bm25.get_top_n(tokenized_query, demon_keys, K_BM25)
    parts = []
    for match in top_matches:
        parts.append(f"Please breakdown the following sentence into independent facts: {match}")
        for fact in demons[match]:
            parts.append(f"- {fact}")
        parts.append("")
    return "\n".join(parts).rstrip("\n")


def parse_atomic_facts(generated_text):
    text = generated_text.replace("<|eot_id|>", "")
    text = re.sub(r"-\s*\n", "", text)
    facts = []
    for line in text.split("\n"):
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
            if facts:
                break
    return facts


# ---------------------------------------------------------------------------
# AFV
# ---------------------------------------------------------------------------

AFV_SYSTEM_INSTRUCT = (
    "You are an annotator that verifies the factuality of a sentence "
    "according to a given source text. You answer only True or False and "
    "provides no further explanations."
)


def build_afv_user_prompt(topic, source, claim):
    definition = f"Answer the question about {topic} based on the given context.\n\n"
    context = f"Title: {topic}\nText: {source.strip()}\n\n"
    definition += context.strip()
    if definition[-1] not in string.punctuation:
        definition += "."
    return f"{definition.strip()}\n\nInput: {claim.strip()} True or False?\nAnswer:"


def parse_afv_label(generated_text):
    answer = generated_text.lower()
    if "true" in answer or "false" in answer:
        if "true" in answer and "false" not in answer:
            is_supported = True
        elif "false" in answer and "true" not in answer:
            is_supported = False
        else:
            is_supported = answer.index("true") > answer.index("false")
    else:
        stripped = answer.translate(str.maketrans("", "", string.punctuation)).split()
        is_supported = all(
            kw not in stripped for kw in ("not", "cannot", "unknown", "information")
        )
    return "SUPPORTED" if is_supported else "NOT_SUPPORTED"


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class HFChatModel:
    def __init__(self, model_id, role_label, use_4bit: bool = False):
        print(f"[{role_label}] loading {model_id} (4-bit={use_4bit}) ...", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"device_map": "auto", "trust_remote_code": True}
        if use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        print(
            f"[{role_label}] loaded in {time.time()-t0:.1f}s · device map: "
            f"{getattr(self.model, 'hf_device_map', 'n/a')}",
            flush=True,
        )

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
        demos_block = build_afg_demos_block(demons, demon_keys, bm25, sent)
        system_prompt = f"{AFG_SYSTEM_INSTRUCT}\n{demos_block}"
        user_prompt = f"Please breakdown the following sentence into independent facts: {sent}"
        out = afg_model.generate(system_prompt, user_prompt, AFG_MAX_NEW_TOKENS)
        all_facts.extend(parse_atomic_facts(out))
    return all_facts


def validate_facts(afv_model, source, facts, topic):
    results = []
    for fact in facts:
        user_prompt = build_afv_user_prompt(topic, source, fact)
        out = afv_model.generate(AFV_SYSTEM_INSTRUCT, user_prompt, AFV_MAX_NEW_TOKENS)
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
    counts = {"SUPPORTED": 0, "NOT_SUPPORTED": 0}
    for v in verified:
        counts[v["label"]] = counts.get(v["label"], 0) + 1

    init_score = counts["SUPPORTED"] / len(facts)
    length_penalty = min(1.0, len(facts) / gamma) if gamma > 0 else 1.0
    final_score = init_score * length_penalty

    return {
        "n_facts": len(facts),
        "n_supported": counts["SUPPORTED"],
        "n_not_supported": counts["NOT_SUPPORTED"],
        "n_contradicted": 0,
        "init_score": init_score,
        "factscore": final_score,
        "verified_facts": verified,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenFActScore over FictionalQA rewriting chains.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV,
                        help=f"Chain CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMOS,
                        help=f"demons.json for AFG few-shot (default: {DEFAULT_DEMOS})")
    parser.add_argument("--afg-model", default=AFG_MODEL_ID,
                        help=f"HF model id for AFG (default: {AFG_MODEL_ID})")
    parser.add_argument("--afv-model", default=AFV_MODEL_ID,
                        help=f"HF model id for AFV (default: {AFV_MODEL_ID})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke-test: only score the first N (step>0) rows.")
    parser.add_argument("--qid", action="append", default=None,
                        help="Restrict evaluation to one or more qid values (repeatable).")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Enable 4-bit NF4 quantization. Default: bfloat16.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Chain CSV not found: {args.input}")
    if not args.demos.exists():
        raise FileNotFoundError(f"demons.json not found: {args.demos}")

    print("=" * 70)
    print("OpenFActScore — source-faithfulness against E_0 (FictionalQA)")
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
    if args.qid:
        to_eval = to_eval[to_eval["qid"].isin(args.qid)]
        if to_eval.empty:
            raise RuntimeError(f"No rows match --qid {args.qid}")
        print(f"*** Filtering to qid in {args.qid}: {len(to_eval)} rows ***")
    if args.limit:
        to_eval = to_eval.head(args.limit)
        print(f"*** SMOKE TEST: limiting to first {args.limit} rows ***")

    out_scores_path = args.input.with_name(args.input.stem + "_openfactscore.csv")
    out_details_path = args.input.with_name(args.input.stem + "_openfactscore_details.csv")
    print(f"Output scores:  {out_scores_path}")
    print(f"Output details: {out_details_path}")

    done_keys = set()
    if out_scores_path.exists():
        prev = pd.read_csv(out_scores_path)
        done_keys = {tuple(row[k] for k in CHAIN_KEYS + ["step"]) for _, row in prev.iterrows()}
        print(f"Resume: {len(done_keys)} (chain, step) rows already scored — will skip them.")

    afg = HFChatModel(args.afg_model, "AFG", use_4bit=args.use_4bit)
    afv = HFChatModel(args.afv_model, "AFV", use_4bit=args.use_4bit)

    # Extract E_0 facts per qid for recall computation
    e0_facts_cache = {}
    done_e0_qids = set()
    if out_details_path.exists():
        prev_details = pd.read_csv(out_details_path)
        done_e0_qids = set(prev_details[prev_details["step"] == 0]["qid"].unique())

    print("\nExtracting E_0 facts (for recall computation)...")
    for chain_id, e0_text in sources.items():
        qid = chain_id[0]
        if qid in e0_facts_cache or qid in done_e0_qids:
            continue
        print(f"  [AFG/E0] {qid} ...", end=" ", flush=True)
        t0 = time.time()
        facts = extract_atomic_facts(afg, e0_text, demons, demon_keys, bm25)
        e0_facts_cache[qid] = facts
        print(f"{len(facts)} facts [{time.time()-t0:.1f}s]", flush=True)
        e0_row = df[(df["qid"] == qid) & (df["step"] == 0)].iloc[0]
        if facts:
            e0_details = pd.DataFrame([
                {**{k: e0_row[k] for k in CHAIN_KEYS},
                 "step": 0, "fact": f, "label": "E0", "raw": ""}
                for f in facts
            ])
            e0_details.to_csv(out_details_path, mode="a",
                              header=not out_details_path.exists(), index=False, encoding="utf-8")

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

        topic = derive_topic(source, row["qid"])
        result = compute_factscore(
            afg, afv, source, row["text"], topic, demons, demon_keys, bm25,
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
