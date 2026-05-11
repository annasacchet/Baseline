"""
Step 2 of the LLM-as-a-judge pipeline (per `exploratory_notes_and_next_steps.md`).

Input: NOT_SUPPORTED claims from OpenFactScore details
       (results/15q/rewriting_chains_15q_openfactscore_details.csv)

For each NOT_SUPPORTED claim, GPT-4o-mini assigns ONE label from a closed
6-category taxonomy. Constructs are:

  evaluative_adjective       — non-grounded value judgement on a noun
                              ("important naval vessel", "iconic actor")
  vacuous_flourish           — non-falsifiable filler with no informational
                              content ("a testament to dedication",
                              "leaving an indelible mark")
  vague_causal_connection    — narrative bridge that suggests a relation
                              without committing to a specific fact
                              ("paying homage to the legacy of...",
                              "reflecting the era's complexity")
  factual_addition_true      — concrete, verifiable claim that is true in
                              the real world but is NOT grounded in E_0
                              (parametric memory leak)
  factual_addition_false     — concrete, verifiable claim that contradicts
                              real-world knowledge (true hallucination)
  other                      — does not fit the above; usually overly short
                              fragments or judge-side artifacts

Per Wang et al. 2023 / G-Eval, the prompt:
  - is in English (judges are better calibrated on English)
  - uses single-construct one-label classification (no free text mixed with
    structured output)
  - has explicit examples for each category to anchor the judge
  - returns JSON only (parseable, no chain-of-thought leak)

Inter-temporal reliability is exposed via --reliability-rerun: re-runs the
classification on a fixed sample with T=0 (deterministic; should agree
~100%) or with T>0 (variability proxy).

Usage
-----
  # Default: 15q elaborate, all NOT_SUPPORTED claims
  python scripts/15q/taxonomized_claims_eval.py

  # Smoke test (10 claims):
  python scripts/15q/taxonomized_claims_eval.py --limit 10

  # Different instruction:
  python scripts/15q/taxonomized_claims_eval.py --instruction-type shorten

  # Reliability re-run on first 100 claims with T=0.7:
  python scripts/15q/taxonomized_claims_eval.py --limit 100 \\
         --reliability-rerun 1 --temperature 0.7

Output (one row per claim):
  qid, group, instruction_type, run, step, fact, et_text_present (bool),
  label, raw_response
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_DETAILS = REPO_ROOT / "results/15q/rewriting_chains_15q_openfactscore_details.csv"
DEFAULT_CHAINS  = REPO_ROOT / "results/15q/rewriting_chains_15q.csv"
DEFAULT_OUT     = REPO_ROOT / "results/15q/elaborate_taxonomized_claims.csv"

LABELS = [
    "evaluative_adjective",
    "vacuous_flourish",
    "vague_causal_connection",
    "factual_addition_true",
    "factual_addition_false",
    "other",
]

JUDGE_PROMPT = """\
You are a strict text annotator. You will read a CLAIM that a language model
added to a rewritten passage. The claim was already flagged as ABSENT from
the source passage (i.e. it is not grounded in the source text). Your job
is NOT to re-check whether it is in the source — that is given. Your job
is to classify the CLAIM ITSELF into ONE of six categories below.

CRITICAL DISTINCTION
--------------------
"Absent from the source" does NOT mean "false in the world".
A claim can be:
  - true in the world AND absent from the source  → factual_addition_true
  - false in the world AND absent from the source → factual_addition_false
  - non-factual (style, vague filler) AND absent  → categories 1, 2, 3
You will judge truth-in-the-world using only your general knowledge.

Labels
------

(1) evaluative_adjective
    A non-grounded value judgement attached to a noun. Not a factual claim.
    Examples:
      - "important naval vessel"
      - "iconic Hollywood actor"
      - "masterfully executed horror film"

(2) vacuous_flourish
    Non-falsifiable filler with no informational content. Emotive or
    summarizing language with no specific factual commitment.
    Examples:
      - "a testament to her dedication and skill"
      - "leaving an indelible mark on history"
      - "captures the essence of the time"

(3) vague_causal_connection
    A narrative bridge that asserts a relationship without committing to a
    specific verifiable fact.
    Examples:
      - "paying homage to the legacy of the Lecter series"
      - "reflecting the era's complexity"
      - "linking him to a long line of influential figures"

(4) factual_addition_true
    A CONCRETE, verifiable factual claim (date, number, name, event,
    relation) that is TRUE in the real world.
    Examples:
      - "Brett Ratner directed Red Dragon (2002)"
      - "Jodie Foster received an Academy Award for The Silence of the Lambs"
      - "Tennessee has an area of about 42,143 square miles"
      - "William Shakespeare's mother was Mary Arden"

(5) factual_addition_false
    A CONCRETE, verifiable factual claim that you are CONFIDENT contradicts
    real-world facts. A genuine hallucination.
    Examples:
      - "Jefferson Memorial Forest spans over 13,000 acres"   [really ~6,500]
      - "Joy Harmon owns a bakery called Aunt Joy's Cakes"    [no such bakery]
      - "Henry III was crowned in 1300"                       [really 1216]

(6) other
    The claim is too short, a fragment, or not actually a claim.

DECISION RULES (read carefully)
-------------------------------
1. Pick EXACTLY ONE label.
2. Order of priority when ambiguous: factual_addition_* > vague_causal_connection
   > vacuous_flourish > evaluative_adjective.
3. For factual_addition_true vs factual_addition_false:
   - DEFAULT to factual_addition_true.
   - Only choose factual_addition_false if you ACTIVELY KNOW the claim is
     wrong. If the claim is about an obscure person/place/event you do
     not recognize, choose factual_addition_true.
   - Plausibility is NOT enough to flag false. You need positive
     contradicting knowledge.
   - Specifically: a claim like "X was born in YYYY" where you do not
     personally know X's birth year → factual_addition_true (you cannot
     verify it is wrong).
4. Output ONLY a JSON object: {{"label": "<label>"}}. No reasoning, no
   explanation, no extra fields.

CLAIM: {claim}
"""


def call_judge(client: OpenAI, claim: str, model: str, temperature: float,
               retries: int = 3) -> tuple[str, str]:
    """Returns (label, raw_response). Falls back to 'other' on parse failure."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=40,
                response_format={"type": "json_object"},
                messages=[{"role": "user",
                           "content": JUDGE_PROMPT.format(claim=claim)}],
            )
            raw = resp.choices[0].message.content
            try:
                obj = json.loads(raw)
                label = obj.get("label", "").strip()
                if label in LABELS:
                    return label, raw
                # accept lowercase / minor variations
                for canon in LABELS:
                    if label.lower() == canon.lower():
                        return canon, raw
                return "other", raw
            except json.JSONDecodeError:
                return "other", raw
        except Exception as e:
            if attempt < retries - 1:
                print(f"  retry {attempt+1} after error: {e}", flush=True)
                time.sleep(2 ** attempt)
            else:
                print(f"  WARNING: skip after {retries} failures: {e}",
                      flush=True)
                return "ERROR", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--chains",  type=Path, default=DEFAULT_CHAINS,
                        help="For attaching the rewrite text (E_t) to the row.")
    parser.add_argument("--instruction-type", default="elaborate",
                        help="Instruction to classify (default: elaborate).")
    parser.add_argument("--out",     type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model",   default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max claims to classify (None = all).")
    parser.add_argument("--reliability-rerun", type=int, default=0,
                        help="If >0, re-classify the first N already-done "
                             "claims to measure reliability. Writes an "
                             "additional _rerun.csv file.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY before running.", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    details = pd.read_csv(args.details)
    target = details[
        (details["instruction_type"] == args.instruction_type)
        & (details["label"] == "NOT_SUPPORTED")
    ].copy()
    print(f"NOT_SUPPORTED claims for {args.instruction_type}: {len(target)}")

    if args.limit:
        target = target.head(args.limit)
        print(f"Limited to first {args.limit}.")

    # Resume support
    done_keys: set = set()
    if args.out.exists() and args.reliability_rerun == 0:
        prev = pd.read_csv(args.out)
        done_keys = set(zip(prev["qid"], prev["instruction_type"],
                            prev["run"], prev["step"], prev["fact"]))
        print(f"Resume: {len(done_keys)} already done.")

    # Reliability rerun: redo the first N rows we've already classified
    rerun_path = args.out.with_name(args.out.stem + "_rerun.csv")
    if args.reliability_rerun > 0:
        if not args.out.exists():
            print("ERROR: cannot rerun — no prior output file.",
                  file=sys.stderr)
            sys.exit(1)
        prev = pd.read_csv(args.out).head(args.reliability_rerun)
        target = prev[["qid", "group", "instruction_type", "run", "step",
                       "fact"]]
        out_path = rerun_path
        done_keys = set()
        print(f"Reliability rerun on {len(target)} claims → {out_path}")
    else:
        out_path = args.out

    rows = []
    n_done = 0
    n_total = len(target)
    t_start = time.time()

    for _, r in target.iterrows():
        key = (r["qid"], r["instruction_type"], r["run"], r["step"],
               r["fact"])
        if key in done_keys:
            n_done += 1
            continue

        label, raw = call_judge(client, str(r["fact"]),
                                args.model, args.temperature)
        rows.append({
            "qid": r["qid"],
            "group": r.get("group", ""),
            "instruction_type": r["instruction_type"],
            "run": r["run"],
            "step": r["step"],
            "fact": r["fact"],
            "label": label,
            "raw_response": raw,
        })
        n_done += 1

        # checkpoint every 25 rows
        if len(rows) >= 25:
            pd.DataFrame(rows).to_csv(
                out_path, mode="a",
                header=not out_path.exists(), index=False,
            )
            elapsed = time.time() - t_start
            eta = (n_total - n_done) * (elapsed / max(n_done, 1))
            done_count = n_done
            print(f"  [{done_count}/{n_total}] ETA {eta/60:.1f} min",
                  flush=True)
            rows = []

    if rows:
        pd.DataFrame(rows).to_csv(
            out_path, mode="a",
            header=not out_path.exists(), index=False,
        )

    print(f"\nDone in {(time.time()-t_start)/60:.1f} min")
    print(f"Saved: {out_path}")

    # Summary
    out = pd.read_csv(out_path)
    print("\n=== Label distribution (overall) ===")
    print(out["label"].value_counts().to_string())
    print("\n=== Label distribution per step ===")
    print(pd.crosstab(out["step"], out["label"]).to_string())
    print("\n=== Label proportion per step (%) ===")
    print((pd.crosstab(out["step"], out["label"], normalize="index") * 100)
          .round(1).to_string())


if __name__ == "__main__":
    main()
