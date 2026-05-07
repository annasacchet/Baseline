"""
LLM-as-a-judge analysis of compression extremes.

Compares the 5 least compressed and 5 most compressed qids at step1,
asking GPT-4o-mini to analyse what changed and whether the rewritten
text is faithful or just paraphrased/hallucinated.

Output:
  results/300q/compression_gpt_analysis.csv
  results/300q/compression_gpt_analysis.md
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
CHAINS_CSV = REPO_ROOT / "results" / "300q" / "rewriting_chains_300q.csv"
OUT_CSV    = REPO_ROOT / "results" / "300q" / "compression_gpt_analysis.csv"
OUT_MD     = REPO_ROOT / "results" / "300q" / "compression_gpt_analysis.md"

MODEL       = "gpt-4o-mini"
TEMPERATURE = 0.0
N_EXTREMES  = 5  # top/bottom N qids by compression ratio

JUDGE_PROMPT = """\
Here are two versions of the same text. The second was produced by a language \
model asked to "elaborate" on the first.

Tell me what changed between them. Be specific and use concrete examples and \
quotes from both texts. I want to understand: what was lost, what was distorted, \
what was added that wasn't in the original, how the tone or style shifted, and \
whether the rewrite actually elaborated or just compressed/paraphrased.

Then give your overall assessment in a JSON with three fields:
- "analysis": your full free-form commentary in Italian (as detailed as you like)
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
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    original=original, rewritten=rewritten
                )}],
            )
            raw = resp.choices[0].message.content
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                import re
                analysis = re.search(r'"analysis"\s*:\s*"(.*?)"(?=\s*,\s*"(?:lost|added)")', raw, re.DOTALL)
                lost  = re.findall(r'"([^"]+)"', re.search(r'"lost"\s*:\s*\[(.*?)\]',  raw, re.DOTALL).group(1) if re.search(r'"lost"\s*:\s*\[(.*?)\]',  raw, re.DOTALL) else "")
                added = re.findall(r'"([^"]+)"', re.search(r'"added"\s*:\s*\[(.*?)\]', raw, re.DOTALL).group(1) if re.search(r'"added"\s*:\s*\[(.*?)\]', raw, re.DOTALL) else "")
                return {"analysis": analysis.group(1) if analysis else raw[:2000], "lost": lost, "added": added}
        except Exception as e:
            if attempt < retries - 1:
                print(f"  retry {attempt+1} after error: {e}", flush=True)
                time.sleep(2 ** attempt)
            else:
                print(f"  WARNING: skipping after {retries} failures: {e}", flush=True)
                return {"analysis": f"ERROR: {e}", "lost": [], "added": []}


def format_md_block(qid: str, group: str, compression: float, result: dict) -> str:
    lines = []
    lines.append(f"## {qid} | {group} | compression={compression:.3f}\n")
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

    chains = pd.read_csv(CHAINS_CSV)

    # Compute compression ratio per qid (mean over all instruction_type and run)
    orig = chains[chains['step'] == 0][['qid', 'instruction_type', 'run', 'n_tokens']].rename(columns={'n_tokens': 'n_tokens_orig'})
    merged = chains.merge(orig, on=['qid', 'instruction_type', 'run'])
    merged['compression_ratio'] = merged['n_tokens'] / merged['n_tokens_orig']
    ratio_by_qid = merged[merged['step'] == 1].groupby('qid')['compression_ratio'].mean().sort_values()

    least_compressed = ratio_by_qid.tail(N_EXTREMES).index.tolist()
    most_compressed  = ratio_by_qid.head(N_EXTREMES).index.tolist()

    print(f"Least compressed (ratio): {dict(zip(least_compressed, ratio_by_qid.tail(N_EXTREMES).round(3)))}")
    print(f"Most compressed  (ratio): {dict(zip(most_compressed,  ratio_by_qid.head(N_EXTREMES).round(3)))}")

    # For each qid, pick run=0, instruction=elaborate, step 0→1
    targets = []
    for qid in least_compressed:
        targets.append((qid, "least_compressed", ratio_by_qid[qid]))
    for qid in most_compressed:
        targets.append((qid, "most_compressed", ratio_by_qid[qid]))

    # Resume
    done = set()
    if OUT_CSV.exists():
        done_df = pd.read_csv(OUT_CSV)
        done = set(zip(done_df["qid"], done_df["group"]))
        print(f"Resume: {len(done)} already done.")

    rows = []
    md_file = open(OUT_MD, "a", encoding="utf-8")
    if OUT_MD.stat().st_size == 0 if OUT_MD.exists() else True:
        md_file.write("# GPT-4o-mini: Compression Extremes Analysis (300q)\n\n")
        md_file.write("## Least compressed (top 5)\n\n")

    t_start = time.time()
    for i, (qid, group, compression) in enumerate(targets):
        if (qid, group) in done:
            continue

        # Separator in MD between groups
        if i == N_EXTREMES and ("most_compressed_header" not in done):
            md_file.write("\n## Most compressed (bottom 5)\n\n")

        orig_text = chains[(chains['qid'] == qid) & (chains['step'] == 0) &
                           (chains['instruction_type'] == 'elaborate') & (chains['run'] == 0)]['text'].values
        rew_text  = chains[(chains['qid'] == qid) & (chains['step'] == 1) &
                           (chains['instruction_type'] == 'elaborate') & (chains['run'] == 0)]['text'].values

        if len(orig_text) == 0 or len(rew_text) == 0:
            print(f"  skip {qid} — missing text", flush=True)
            continue

        print(f"[{i+1}/{len(targets)}] {qid} ({group}, ratio={compression:.3f}) ...", end=" ", flush=True)
        t0 = time.time()
        result = call_gpt(client, orig_text[0], rew_text[0])
        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s | lost={len(result.get('lost',[]))} added={len(result.get('added',[]))}", flush=True)

        md_file.write(format_md_block(qid, group, compression, result))
        md_file.flush()

        rows.append({
            "qid":         qid,
            "group":       group,
            "compression": round(compression, 3),
            "n_lost":      len(result.get("lost", [])),
            "n_added":     len(result.get("added", [])),
            "lost":        " | ".join(result.get("lost", [])),
            "added":       " | ".join(result.get("added", [])),
            "analysis":    result.get("analysis", ""),
        })

        new_df = pd.DataFrame(rows)
        write_header = not OUT_CSV.exists()
        new_df.to_csv(OUT_CSV, mode="a", header=write_header, index=False)
        rows = []

    md_file.close()
    elapsed_total = time.time() - t_start
    print(f"\nDone in {elapsed_total:.1f}s.")
    print(f"CSV: {OUT_CSV}")
    print(f"MD:  {OUT_MD}")


if __name__ == "__main__":
    main()
