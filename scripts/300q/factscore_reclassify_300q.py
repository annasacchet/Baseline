"""
Fine-grained reclassification of NOT_SUPPORTED facts using Gemma-2-27B-Instruct.

Input:  rewriting_chains_300q_openfactscore_details.csv  (OFS binary labels)
        rewriting_chains_300q.csv                         (chain texts, for E_0 source)

Takes every fact labelled NOT_SUPPORTED and reclassifies it into one of:
  SUPPORTED | CONTRADICTION | INVENTED | DISTORTED | UNVERIFIABLE

Output:
  rewriting_chains_300q_reclassified.csv
    qid, group, instruction_type, run, step, fact, label, reason, evidence_match
    (original_label column kept for reference)
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DETAILS = REPO_ROOT / "results" / "300q" / "rewriting_chains_300q_openfactscore_details.csv"
DEFAULT_CHAINS  = REPO_ROOT / "results" / "300q" / "rewriting_chains_300q.csv"

MODEL_ID = "google/gemma-2-27b-it"
MAX_NEW_TOKENS = 256

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

PROMPT_TEMPLATE = """\

Your task is to classify atomic claims extracted from a rewritten text by comparing them ONLY against a provided ground truth text.

You must NOT use external knowledge. Only the ground truth is valid evidence.

────────────────────────────────────────
INPUTS
────────────────────────────────────────

GROUND TRUTH:
{ground_truth}

ATOMIC CLAIM:
{atomic_claim}

────────────────────────────────────────
TASK
────────────────────────────────────────

This claim has already been judged as NOT supported by the ground truth.
Your job is to classify WHY it is not supported, into EXACTLY ONE label:

1. CONTRADICTION
The claim directly contradicts the ground truth. Both cannot be true simultaneously.

2. INVENTED
The claim introduces information not present in the ground truth and cannot be mapped to any fact in it.

3. DISTORTED
The claim is partially related to the ground truth but modifies meaning, such as:
- wrong entity
- wrong number/date
- changed relationship
- altered factual meaning

4. UNVERIFIABLE
The claim cannot be verified or rejected using only the ground truth.

────────────────────────────────────────
DECISION RULES (VERY IMPORTANT)
────────────────────────────────────────

Follow these priority rules strictly:

1. If the claim conflicts with ground truth → CONTRADICTION (highest priority over INVENTED)

2. If the claim partially overlaps but changes details → DISTORTED (preferred over INVENTED)

3. If the claim has no overlap with ground truth → INVENTED

4. If ground truth does not contain enough information to decide → UNVERIFIABLE

────────────────────────────────────────
DEFINITION OF ATOMICITY
────────────────────────────────────────

- Each claim should represent a single fact.
- Do not split or merge claims.
- Evaluate only the given claim as-is.

────────────────────────────────────────
OUTPUT FORMAT (STRICT JSON)
────────────────────────────────────────

Return ONLY a valid JSON object:

{{
  "label": "CONTRADICTION | INVENTED | DISTORTED | UNVERIFIABLE",
  "reason": "short explanation grounded in the text",
  "evidence_match": "brief mention of relevant part of ground truth or 'none'"
}}

────────────────────────────────────────
STRICT CONSTRAINTS
────────────────────────────────────────

- Do NOT use world knowledge.
- Do NOT guess missing facts.
- Be conservative: prefer DISTORTED over INVENTED when overlap exists.
- Prefer CONTRADICTION over INVENTED when conflict exists.
- Keep reasoning short and evidence-based.\
"""

VALID_LABELS = {"CONTRADICTION", "INVENTED", "DISTORTED", "UNVERIFIABLE"}


def parse_response(text):
    """Extract JSON from model output; return (label, reason, evidence_match) or None."""
    # Try to find a JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    label = obj.get("label", "").strip().upper()
    if label not in VALID_LABELS:
        # Try to salvage if label is embedded in a longer string
        for v in VALID_LABELS:
            if v in label:
                label = v
                break
        else:
            return None
    return {
        "label": label,
        "reason": str(obj.get("reason", "")).strip(),
        "evidence_match": str(obj.get("evidence_match", "")).strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Reclassify NOT_SUPPORTED facts with Gemma-2-27B.")
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--chains",  type=Path, default=DEFAULT_CHAINS)
    parser.add_argument("--model",   default=MODEL_ID)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--limit",   type=int, default=None, help="Smoke-test: only process first N facts.")
    parser.add_argument("--qid",     action="append", default=None)
    args = parser.parse_args()

    for p in (args.details, args.chains):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    print("=" * 70)
    print("Fine-grained reclassification — NOT_SUPPORTED facts")
    print(f"  Model: {args.model}  4-bit={args.use_4bit}")
    print("=" * 70)

    # Load E_0 source texts (one per qid)
    chains = pd.read_csv(args.chains)
    e0_texts = (
        chains[chains["step"] == 0][["qid", "text"]]
        .drop_duplicates("qid")
        .set_index("qid")["text"]
        .to_dict()
    )
    print(f"Loaded {len(e0_texts)} E_0 source texts")

    # Load NOT_SUPPORTED facts
    details = pd.read_csv(args.details)
    to_classify = details[details["label"] == "NOT_SUPPORTED"].copy().reset_index(drop=True)
    if args.qid:
        to_classify = to_classify[to_classify["qid"].isin(args.qid)].reset_index(drop=True)
    if args.limit:
        to_classify = to_classify.head(args.limit)
        print(f"*** SMOKE TEST: first {args.limit} facts ***")
    print(f"Facts to reclassify: {len(to_classify)}")

    out_path = args.details.with_name(args.details.stem.replace("_openfactscore_details", "") + "_reclassified.csv")
    print(f"Output: {out_path}")

    # Resumability: skip already-done (qid, group, instruction_type, run, step, fact)
    done_keys = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done_keys = {
            (r["qid"], r["group"], r["instruction_type"], r["run"], r["step"], r["fact"])
            for _, r in prev.iterrows()
        }
        print(f"Resume: {len(done_keys)} facts already classified.")

    # Load model
    print(f"\nLoading {args.model} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kwargs = {"device_map": "auto"}
    if args.use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    total = len(to_classify)
    t_start = time.time()
    n_done = 0

    for i, row in to_classify.iterrows():
        key = (row["qid"], row["group"], row["instruction_type"], row["run"], row["step"], row["fact"])
        if key in done_keys:
            continue

        e0 = e0_texts.get(row["qid"], "")
        prompt = PROMPT_TEMPLATE.format(ground_truth=e0, atomic_claim=row["fact"])

        messages = [{"role": "user", "content": prompt}]
        text_in = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_in, return_tensors="pt").to(model.device)

        t_row = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        elapsed = time.time() - t_row

        parsed = parse_response(generated)
        if parsed is None:
            label, reason, evidence_match = "PARSE_ERROR", generated[:120], ""
        else:
            label, reason, evidence_match = parsed["label"], parsed["reason"], parsed["evidence_match"]

        result_row = pd.DataFrame([{
            **{k: row[k] for k in CHAIN_KEYS},
            "step": row["step"],
            "fact": row["fact"],
            "original_label": row["label"],
            "label": label,
            "reason": reason,
            "evidence_match": evidence_match,
        }])
        result_row.to_csv(out_path, mode="a", header=not out_path.exists(), index=False, encoding="utf-8")

        n_done += 1
        print(f"[{n_done}/{total}] {row['qid']} step{int(row['step'])} → {label}  [{elapsed:.1f}s]", flush=True)

        if n_done % 50 == 0:
            avg = (time.time() - t_start) / n_done
            print(f"   ETA: {(total - n_done) * avg / 3600:.1f}h  (avg {avg:.1f}s/fact)", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nDone. {n_done} facts in {elapsed_total/3600:.1f}h")
    print(f"Saved: {out_path}")

    print("\n" + "=" * 70)
    print("Label distribution")
    print("=" * 70)
    out = pd.read_csv(out_path)
    print(out["label"].value_counts())
    print()
    print("By step:")
    print(out.groupby("step")["label"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
