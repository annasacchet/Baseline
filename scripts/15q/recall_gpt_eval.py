"""
recall_gpt_eval.py — OFS recall via GPT-4o-mini as AFV.

Reads E_0 facts from the OFS details CSV (label='E0'), then for each
fact verifies whether it is supported by E_t (t=1,2,3) using GPT-4o-mini.

Recall = facts from E_0 that survive in E_t / total E_0 facts

Usage:
  python scripts/15q/recall_gpt_eval.py \
      --chains  results/300q/rewriting_chains_300q.csv \
      --details results/300q/rewriting_chains_300q_openfactscore_details.csv \
      --limit 500
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

AFV_PROMPT = """\
You are a fact verification assistant. Given a source text and a claim, \
answer only True or False — no explanations.

Source text:
{source}

Claim: {claim} True or False?
Answer:"""


def verify_fact(client: OpenAI, source: str, fact: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=5,
                messages=[{"role": "user", "content": AFV_PROMPT.format(
                    source=source, claim=fact
                )}],
            )
            answer = resp.choices[0].message.content.strip().lower()
            if "true" in answer:
                return "SURVIVED"
            if "false" in answer:
                return "LOST"
            return "SURVIVED"  # default fallback
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  WARNING: {e}", flush=True)
                return "ERROR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains",  type=Path,
                        default=REPO_ROOT / "results/300q/rewriting_chains_300q.csv")
    parser.add_argument("--details", type=Path,
                        default=REPO_ROOT / "results/300q/rewriting_chains_300q_openfactscore_details.csv")
    parser.add_argument("--limit",   type=int, default=500,
                        help="Max number of E_0 facts to verify (default: 500)")
    parser.add_argument("--output",  type=Path, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY before running.", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    chains  = pd.read_csv(args.chains)
    details = pd.read_csv(args.details)

    # E_0 facts — restrict to qids present in the chains CSV,
    # sample evenly across instruction_types to avoid imbalance
    valid_qids = set(chains['qid'].unique())
    e0_all = details[(details['step'] == 0) & (details['qid'].isin(valid_qids))][
        ['qid','instruction_type','run','fact']
    ].drop_duplicates()

    if args.limit:
        per_instr = args.limit // e0_all['instruction_type'].nunique()
        e0_facts  = (e0_all.groupby('instruction_type', group_keys=False)
                           .apply(lambda x: x.head(per_instr))
                           .reset_index(drop=True))
    else:
        e0_facts = e0_all
    print(f"E_0 facts to verify: {len(e0_facts)} (limit={args.limit})")

    # E_t texts indexed by (qid, instruction_type, run, step)
    et_texts = chains[chains['step'] > 0].set_index(
        ['qid','instruction_type','run','step']
    )['text'].to_dict()

    out_path = args.output or args.details.with_name(
        args.details.stem.replace('_details', '') + "_recall_gpt.csv"
    )
    print(f"Output: {out_path}")

    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(zip(prev['qid'], prev['instruction_type'], prev['run'],
                       prev['step'], prev['fact']))
        print(f"Resume: {len(done)} already done.")

    # Build list of tasks to run
    tasks = []
    for _, frow in e0_facts.iterrows():
        qid   = frow['qid']
        instr = frow['instruction_type']
        run   = frow['run']
        fact  = frow['fact']
        for step in [1, 2, 3]:
            key = (qid, instr, run, step, fact)
            if key in done:
                continue
            et_text = et_texts.get((qid, instr, run, step))
            if et_text is None:
                continue
            tasks.append((qid, instr, run, step, fact, et_text))

    total   = len(tasks)
    rows    = []
    n       = 0
    t_start = time.time()
    print(f"Tasks to run: {total}", flush=True)

    for task in tasks:
        qid, instr, run, step, fact, et_text = task
        label = verify_fact(client, et_text, fact)
        rows.append({"qid": qid, "instruction_type": instr, "run": run,
                     "step": step, "fact": fact, "label": label})
        n += 1
        if n % 50 == 0:
            pd.DataFrame(rows).to_csv(out_path, mode="a",
                header=not out_path.exists(), index=False)
            rows = []
            elapsed = time.time() - t_start
            eta = (total - n) * (elapsed / max(n, 1))
            print(f"  [{n}/{total}] ETA: {eta/60:.1f} min", flush=True)

    if rows:
        pd.DataFrame(rows).to_csv(out_path, mode="a",
            header=not out_path.exists(), index=False)

    print(f"\nDone in {(time.time()-t_start)/60:.1f} min")
    print(f"Saved: {out_path}")

    # Summary
    out = pd.read_csv(out_path)
    out['survived'] = (out['label'] == 'SURVIVED').astype(float)
    print("\n=== Recall per instruction e step ===")
    pivot = out.groupby(['instruction_type','step'])['survived'].mean().unstack()
    print(pivot.round(3))


if __name__ == "__main__":
    main()
