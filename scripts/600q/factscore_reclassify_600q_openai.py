"""
Reclassify NOT_SUPPORTED facts using an OpenAI model as judge.

Counterpart to factscore_reclassify_300q.py (which uses Gemma-2-27B locally).
The point of this script is to *cross-check* the Gemma judge with an external
model on the same claims — so we can separate:

  - genuine rewriting errors,
  - false positives of the AFV (Gemma-3-4B) that mislabelled supported claims,
  - errors of the local judge itself.

Key differences vs the Gemma reclassify script:

  - Uses OpenAIChat (scripts/smoke_openai/openai_chat.py) — needs OPENAI_API_KEY.
  - Adds a 5th SUPPORTED category: lets the OpenAI judge reroute claims that
    the AFV originally mis-labelled as NOT_SUPPORTED but are actually fine.
  - Defaults to gpt-4o-mini (cheap, deterministic at temperature=0).
  - Single-qid friendly: pass --qid; otherwise restrict with --limit.

Output: <details>.with_suffix replaced → _reclassified_openai.csv
Columns: qid, group, instruction_type, run, step, fact, original_label,
         label, reason, evidence_match
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "smoke_openai"))
from openai_chat import OpenAIChat  # noqa: E402

DEFAULT_DETAILS = REPO_ROOT / "results" / "600q" / "rewriting_chains_musique_600q_openfactscore_details.csv"
DEFAULT_CHAINS  = REPO_ROOT / "results" / "600q" / "rewriting_chains_musique_600q.csv"
DEFAULT_QID     = "2hop__14092_8311"

DEFAULT_MODEL = "gpt-4o-mini"
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

This claim was previously judged as NOT supported by a smaller verifier model,
but that verifier is known to make mistakes. Your job is to issue an
independent verdict.

Classify the claim into EXACTLY ONE of the following labels:

1. SUPPORTED
The claim is in fact supported by the ground truth — either stated explicitly
or derivable via a single trivial inference (e.g. "X is the daughter of Y;
Y is the daughter of Z" → "X is the granddaughter of Z"). Use this label when
the previous verifier was wrong.

2. CONTRADICTION
The claim directly contradicts the ground truth. Both cannot be true
simultaneously.

3. INVENTED
The claim introduces information not present in the ground truth and cannot
be mapped to any fact in it.

4. DISTORTED
The claim is partially related to the ground truth but modifies meaning, such
as: wrong entity, wrong number/date, changed relationship, altered factual
meaning.

5. UNVERIFIABLE
The claim cannot be verified or rejected using only the ground truth.

────────────────────────────────────────
DECISION RULES (VERY IMPORTANT)
────────────────────────────────────────

Priority order:

1. If the claim is in fact supported (or a trivial inference of) the ground
   truth → SUPPORTED.

2. If the claim conflicts with ground truth → CONTRADICTION (highest priority
   over INVENTED).

3. If the claim partially overlaps but changes details → DISTORTED (preferred
   over INVENTED).

4. If the claim has no overlap with ground truth → INVENTED.

5. If ground truth does not contain enough information to decide →
   UNVERIFIABLE.

────────────────────────────────────────
OUTPUT FORMAT (STRICT JSON)
────────────────────────────────────────

Return ONLY a valid JSON object:

{{
  "label": "SUPPORTED | CONTRADICTION | INVENTED | DISTORTED | UNVERIFIABLE",
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
- Prefer SUPPORTED when the claim is genuinely entailed.
- Keep reasoning short and evidence-based.\
"""

VALID_LABELS = {"SUPPORTED", "CONTRADICTION", "INVENTED", "DISTORTED", "UNVERIFIABLE"}


def parse_response(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    label = str(obj.get("label", "")).strip().upper()
    if label not in VALID_LABELS:
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
    parser = argparse.ArgumentParser(description="Reclassify NOT_SUPPORTED facts with an OpenAI judge.")
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--chains",  type=Path, default=DEFAULT_CHAINS)
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--limit",   type=int, default=None, help="Only process first N facts.")
    parser.add_argument("--qid",     action="append", default=None,
                        help=f"Restrict to qid (repeatable). Default: --qid {DEFAULT_QID}")
    parser.add_argument("--out",     type=Path, default=None,
                        help="Output CSV. Default: <details>_reclassified_openai.csv")
    args = parser.parse_args()

    qids = args.qid or [DEFAULT_QID]

    for p in (args.details, args.chains):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    print("=" * 70)
    print("Fine-grained reclassification — NOT_SUPPORTED facts (OpenAI judge, 600q)")
    print(f"  Model: {args.model}")
    print(f"  QIDs:  {qids}")
    print("=" * 70)

    chains = pd.read_csv(args.chains, low_memory=False)
    e0_texts = (
        chains[chains["step"] == 0][["qid", "text"]]
        .drop_duplicates("qid")
        .set_index("qid")["text"]
        .to_dict()
    )
    print(f"Loaded {len(e0_texts)} E_0 source texts")

    details = pd.read_csv(args.details, low_memory=False)
    to_classify = details[(details["label"] == "NOT_SUPPORTED") & (details["qid"].isin(qids))].copy()
    to_classify = to_classify.reset_index(drop=True)
    if args.limit:
        to_classify = to_classify.head(args.limit)
        print(f"*** SMOKE: first {args.limit} facts ***")
    print(f"Facts to reclassify: {len(to_classify)}")

    out_path = args.out or args.details.with_name(
        args.details.stem.replace("_openfactscore_details", "") + "_reclassified_openai.csv"
    )
    print(f"Output: {out_path}")

    # Resume
    done_keys = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done_keys = {
            (r["qid"], r["group"], r["instruction_type"], r["run"], r["step"], r["fact"])
            for _, r in prev.iterrows()
        }
        print(f"Resume: {len(done_keys)} facts already classified.")

    judge = OpenAIChat(model=args.model, role_label="reclassify-judge", temperature=0.0)

    t_start = time.time()
    n_done = 0
    total = len(to_classify)

    for _, row in to_classify.iterrows():
        key = (row["qid"], row["group"], row["instruction_type"], row["run"], row["step"], row["fact"])
        if key in done_keys:
            continue

        e0 = e0_texts.get(row["qid"], "")
        prompt = PROMPT_TEMPLATE.format(ground_truth=e0, atomic_claim=row["fact"])

        t_row = time.time()
        generated = judge.complete(prompt, max_tokens=MAX_NEW_TOKENS)
        elapsed = time.time() - t_row

        parsed = parse_response(generated)
        if parsed is None:
            label, reason, evidence_match = "PARSE_ERROR", generated[:200], ""
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
        print(f"[{n_done}/{total}] {row['qid']} step{int(row['step'])} {row['instruction_type'][:5]} → {label}  [{elapsed:.1f}s]", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nDone. {n_done} facts in {elapsed_total/60:.1f} min")
    print(f"Saved: {out_path}")

    print("\n" + "=" * 70)
    print("Label distribution")
    print("=" * 70)
    out = pd.read_csv(out_path)
    print(out["label"].value_counts())
    print("\nBy step:")
    print(out.groupby("step")["label"].value_counts().unstack(fill_value=0))
    print("\nBy instruction:")
    print(out.groupby("instruction_type")["label"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
