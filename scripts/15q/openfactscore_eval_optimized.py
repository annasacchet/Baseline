"""
OpenFActScore-style evaluation — OPTIMIZED GPU version.

Same pipeline, prompts, parsing, and output schema as
scripts/15q/openfactscore_eval.py. The ONLY differences are how the model
talks to the GPU:

  * AFG and AFV now use batched generation (padding_side='left' +
    padding=True), instead of one prompt at a time.
  * KV-cache is explicitly enabled, and torch.inference_mode() replaces
    torch.no_grad() for slightly less overhead.
  * Two CLI flags --afg-batch / --afv-batch control batch sizes.
    AFV emits ≤8 tokens, so its batch can be large (default 32).
    AFG generates up to 256 tokens, so its batch is more modest (default 8).

Everything else (models, prompts, demos, schema, resume logic, CSV layout)
is identical to the non-optimized script. Outputs go to the same files.
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
DEFAULT_CSV = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q.csv"
DEFAULT_DEMOS = REPO_ROOT / "data" / "demons.json"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

AFG_MODEL_ID = "allenai/OLMo-2-1124-7B-SFT"
AFV_MODEL_ID = "google/gemma-3-4b-it"

K_BM25 = 1
AFG_MAX_NEW_TOKENS = 256
AFV_MAX_NEW_TOKENS = 8

DEFAULT_AFG_BATCH = 8
DEFAULT_AFV_BATCH = 32

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Sentence splitting (verbatim from upstream factscore/atomic_facts.py)
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
# Model wrapper with BATCHED generation
# ---------------------------------------------------------------------------

class HFChatModel:
    def __init__(self, model_id, role_label, use_4bit: bool = False):
        print(f"[{role_label}] loading {model_id} (4-bit={use_4bit}) ...", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only batched generation requires left padding.
        self.tokenizer.padding_side = "left"
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
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True
        print(
            f"[{role_label}] loaded in {time.time()-t0:.1f}s · device map: "
            f"{getattr(self.model, 'hf_device_map', 'n/a')}",
            flush=True,
        )

    def _format_one(self, system_prompt, user_prompt):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        sys_block = f"{system_prompt}\n\n" if system_prompt else ""
        return f"{sys_block}{user_prompt}"

    @torch.inference_mode()
    def generate_batch(self, pairs, max_new_tokens):
        """Batched generation. pairs = list of (system_prompt, user_prompt)."""
        if not pairs:
            return []
        texts = [self._format_one(sp, up) for sp, up in pairs]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.model.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return [d.strip() for d in decoded]


# ---------------------------------------------------------------------------
# Pipeline (batched)
# ---------------------------------------------------------------------------

def extract_atomic_facts_batched(afg_model, text, demons, demon_keys, bm25, batch_size):
    sentences = sentences_from_text(text)
    if not sentences:
        return []
    pairs = []
    for sent in sentences:
        demos_block = build_afg_demos_block(demons, demon_keys, bm25, sent)
        system_prompt = f"{AFG_SYSTEM_INSTRUCT}\n{demos_block}"
        user_prompt = f"Please breakdown the following sentence into independent facts: {sent}"
        pairs.append((system_prompt, user_prompt))

    all_facts = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        outs = afg_model.generate_batch(batch, AFG_MAX_NEW_TOKENS)
        for out in outs:
            all_facts.extend(parse_atomic_facts(out))
    return all_facts


def validate_facts_batched(afv_model, source, facts, topic, batch_size):
    if not facts:
        return []
    pairs = [
        (AFV_SYSTEM_INSTRUCT, build_afv_user_prompt(topic, source, fact))
        for fact in facts
    ]
    results = []
    for i in range(0, len(pairs), batch_size):
        chunk_facts = facts[i:i + batch_size]
        chunk_pairs = pairs[i:i + batch_size]
        outs = afv_model.generate_batch(chunk_pairs, AFV_MAX_NEW_TOKENS)
        for fact, out in zip(chunk_facts, outs):
            results.append({"fact": fact, "label": parse_afv_label(out), "raw": out})
    return results


def compute_factscore_batched(
    afg_model, afv_model, source, generated, topic,
    demons, demon_keys, bm25,
    afg_batch, afv_batch, gamma=10,
):
    facts = extract_atomic_facts_batched(
        afg_model, generated, demons, demon_keys, bm25, afg_batch,
    )
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

    verified = validate_facts_batched(afv_model, source, facts, topic, afv_batch)
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
    parser = argparse.ArgumentParser(description="OpenFActScore (OLMo + Gemma) — optimized batched GPU version.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--afg-model", default=AFG_MODEL_ID)
    parser.add_argument("--afv-model", default=AFV_MODEL_ID)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qid", action="append", default=None)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--afg-batch", type=int, default=DEFAULT_AFG_BATCH,
                        help=f"AFG batch size (default: {DEFAULT_AFG_BATCH}).")
    parser.add_argument("--afv-batch", type=int, default=DEFAULT_AFV_BATCH,
                        help=f"AFV batch size (default: {DEFAULT_AFV_BATCH}).")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Chain CSV not found: {args.input}")
    if not args.demos.exists():
        raise FileNotFoundError(f"demons.json not found: {args.demos}")

    print("=" * 70)
    print("OpenFActScore (OPTIMIZED) — source-faithfulness vs E_0")
    print(f"  AFG: {args.afg_model}  batch={args.afg_batch}")
    print(f"  AFV: {args.afv_model}  batch={args.afv_batch}")
    print("=" * 70)

    with open(args.demos) as f:
        demons = json.load(f)
    demon_keys = list(demons.keys())
    bm25 = BM25Okapi([doc.split(" ") for doc in demon_keys])
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
        print(f"Resume: {len(done_keys)} (chain, step) rows already scored.")

    afg = HFChatModel(args.afg_model, "AFG", use_4bit=args.use_4bit)
    afv = HFChatModel(args.afv_model, "AFV", use_4bit=args.use_4bit)

    # E_0 facts (one extraction per qid)
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
        facts = extract_atomic_facts_batched(
            afg, e0_text, demons, demon_keys, bm25, args.afg_batch,
        )
        e0_facts_cache[qid] = facts
        print(f"{len(facts)} facts [{time.time()-t0:.1f}s]", flush=True)
        e0_row = df[(df["qid"] == qid) & (df["step"] == 0)].iloc[0]
        if facts:
            pd.DataFrame([
                {**{k: e0_row[k] for k in CHAIN_KEYS},
                 "step": 0, "fact": f, "label": "E0", "raw": ""}
                for f in facts
            ]).to_csv(out_details_path, mode="a",
                      header=not out_details_path.exists(),
                      index=False, encoding="utf-8")

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

        result = compute_factscore_batched(
            afg, afv, source, row["text"], row["qid"],
            demons, demon_keys, bm25,
            afg_batch=args.afg_batch, afv_batch=args.afv_batch,
        )
        elapsed = time.time() - t0

        pd.DataFrame([{
            **{k: row[k] for k in CHAIN_KEYS},
            "step": int(row["step"]),
            "instruction_used": row.get("instruction_used"),
            **{k: v for k, v in result.items() if k != "verified_facts"},
        }]).to_csv(out_scores_path, mode="a",
                   header=not out_scores_path.exists(),
                   index=False, encoding="utf-8")

        if result["verified_facts"]:
            pd.DataFrame([
                {**{k: row[k] for k in CHAIN_KEYS},
                 "step": int(row["step"]),
                 "fact": vf["fact"], "label": vf["label"], "raw": vf["raw"]}
                for vf in result["verified_facts"]
            ]).to_csv(out_details_path, mode="a",
                      header=not out_details_path.exists(),
                      index=False, encoding="utf-8")

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
