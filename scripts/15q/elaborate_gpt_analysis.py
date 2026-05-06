"""
GPT-4o-mini analysis of what is lost/added between rewriting steps for the 'elaborate' instruction.

For each (qid, run), compares:
  - consecutive pairs: (0→1), (1→2), (2→3)
  - overall:           (0→3)

Output:
  - results/15q/elaborate_gpt_analysis.csv  (structured, one row per comparison)
  - results/15q/elaborate_gpt_analysis.md   (human-readable report)
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHAINS_CSV = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q.csv"
OUT_CSV    = REPO_ROOT / "results" / "15q" / "elaborate_gpt_analysis.csv"
OUT_MD     = REPO_ROOT / "results" / "15q" / "elaborate_gpt_analysis.md"

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0

JUDGE_PROMPT = """\
Here are two versions of the same text. The second was produced by a language model asked to "elaborate" on the first.

Tell me what changed between them. Be specific and use concrete examples and quotes from both texts. \
I want to understand: what was lost, what was distorted, what was added that wasn't in the original, \
how the tone or style shifted, and whether the rewrite actually elaborated or just compressed/paraphrased.

Then give your overall assessment in a JSON with two fields:
- "analysis": your full free-form commentary (as detailed as you like)
- "lost": a list of specific factual claims that disappeared or were distorted, each as a short string
- "added": a list of claims introduced that are not grounded in the original, each as a short string

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
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt < retries - 1:
                print(f"  retry {attempt+1} after error: {e}", flush=True)
                time.sleep(2 ** attempt)
            else:
                raise


def format_md_block(qid: str, run: int, label: str, result: dict) -> str:
    lines = []
    lines.append(f"## {qid} | run {run} | {label}\n")

    analysis = result.get("analysis", "")
    if analysis:
        lines.append(f"{analysis}\n")

    lost = result.get("lost", [])
    if lost:
        lines.append(f"**Lost ({len(lost)}):** " + " | ".join(f"`{c}`" for c in lost) + "\n")

    added = result.get("added", [])
    if added:
        lines.append(f"**Added ({len(added)}):** " + " | ".join(f"`{c}`" for c in added) + "\n")

    lines.append("---\n")
    return "\n".join(lines)


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY before running.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(CHAINS_CSV)
    elab = df[df["instruction_type"] == "elaborate"].copy()

    pivot = elab.pivot_table(index=["qid", "run"], columns="step", values="text", aggfunc="first")

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

    md_file = open(OUT_MD, "a", encoding="utf-8")
    if OUT_MD.stat().st_size == 0 if OUT_MD.exists() else True:
        md_file.write("# GPT-4o-mini Analysis: Elaborate Rewriting Chains\n\n")

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

            # Write markdown block immediately
            md_file.write(format_md_block(qid, run, label, result))
            md_file.flush()

            lost_list  = result.get("lost", [])
            added_list = result.get("added", [])
            rows.append({
                "qid":        qid,
                "run":        run,
                "comparison": label,
                "src_step":   src_step,
                "tgt_step":   tgt_step,
                "n_lost":     len(lost_list),
                "n_added":    len(added_list),
                "lost":       " | ".join(lost_list),
                "added":      " | ".join(added_list),
                "analysis":   result.get("analysis", ""),
            })
            n_done += 1

            new_df = pd.DataFrame(rows)
            write_header = not OUT_CSV.exists()
            new_df.to_csv(OUT_CSV, mode="a", header=write_header, index=False)
            rows = []

    md_file.close()
    elapsed_total = time.time() - t_start
    print(f"\nDone in {elapsed_total/60:.1f} min.", flush=True)
    print(f"CSV:      {OUT_CSV}", flush=True)
    print(f"Report:   {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
