"""
GPT-4o analysis of what is lost/added between rewriting steps for the 'elaborate' instruction.

For each (qid, run), compares:
  - consecutive pairs: (0→1), (1→2), (2→3)
  - overall:           (0→3)

Output: results/15q/elaborate_gpt_analysis.csv
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHAINS_CSV = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q.csv"
OUT_CSV    = REPO_ROOT / "results" / "15q" / "elaborate_gpt_analysis.csv"

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0

JUDGE_PROMPT = """\
You are a fact-checking assistant. You are given two versions of a text: an ORIGINAL and a REWRITTEN version.

Your task:
1. List all factual claims present in ORIGINAL but missing or distorted in REWRITTEN (LOST).
2. List all factual claims present in REWRITTEN but not grounded in ORIGINAL (ADDED).
3. Give a brief overall summary (1-2 sentences) of what changed.

Respond in this exact JSON format:
{{
  "lost": ["claim 1", "claim 2", ...],
  "added": ["claim 1", "claim 2", ...],
  "summary": "..."
}}

ORIGINAL:
{original}

REWRITTEN:
{rewritten}"""


def call_gpt(client: OpenAI, original: str, rewritten: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "user", "content": JUDGE_PROMPT.format(
                        original=original, rewritten=rewritten
                    )}
                ],
            )
            import json
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt < retries - 1:
                print(f"  retry {attempt+1} after error: {e}", flush=True)
                time.sleep(2 ** attempt)
            else:
                raise


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY before running.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(CHAINS_CSV)
    elab = df[df["instruction_type"] == "elaborate"].copy()

    # Build pivot: rows = (qid, run), cols = step 0..3
    pivot = elab.pivot_table(index=["qid", "run"], columns="step", values="text", aggfunc="first")

    # Resume: skip already-done (qid, run, comparison)
    done = set()
    if OUT_CSV.exists():
        done_df = pd.read_csv(OUT_CSV)
        done = set(zip(done_df["qid"], done_df["run"], done_df["comparison"]))
        print(f"Resume: {len(done)} comparisons already done.", flush=True)

    comparisons = [(0, 1), (1, 2), (2, 3), (0, 3)]
    rows = []

    total = len(pivot) * len(comparisons)
    n_done = 0
    t_start = time.time()

    for (qid, run), texts in pivot.iterrows():
        for (src_step, tgt_step) in comparisons:
            label = f"{src_step}→{tgt_step}"
            if (qid, run, label) in done:
                n_done += 1
                continue

            original  = texts[src_step]
            rewritten = texts[tgt_step]

            if pd.isna(original) or pd.isna(rewritten):
                print(f"  skip {qid} run{run} {label} — missing text", flush=True)
                n_done += 1
                continue

            print(f"[{n_done+1}/{total}] {qid} run{run} {label} ...", end=" ", flush=True)
            t0 = time.time()
            result = call_gpt(client, original, rewritten)
            elapsed = time.time() - t0
            print(f"{elapsed:.1f}s | lost={len(result.get('lost',[]))} added={len(result.get('added',[]))}", flush=True)

            rows.append({
                "qid":        qid,
                "run":        run,
                "comparison": label,
                "src_step":   src_step,
                "tgt_step":   tgt_step,
                "n_lost":     len(result.get("lost", [])),
                "n_added":    len(result.get("added", [])),
                "lost":       " | ".join(result.get("lost", [])),
                "added":      " | ".join(result.get("added", [])),
                "summary":    result.get("summary", ""),
            })
            n_done += 1

            # Append incrementally so progress is saved
            if rows:
                new_df = pd.DataFrame(rows)
                write_header = not OUT_CSV.exists()
                new_df.to_csv(OUT_CSV, mode="a", header=write_header, index=False)
                rows = []

    elapsed_total = time.time() - t_start
    print(f"\nDone in {elapsed_total/60:.1f} min. Output: {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
