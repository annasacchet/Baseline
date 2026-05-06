"""
Sign tests on rewriting chain metrics for all instruction types and groups.

Tests:
  1. FactScore step1 > step3        (faithfulness degrades cumulatively?)
  2. BERTScore baseline step1 > step3  (drift from original grows?)
  3. Answer F1 step0 > step1        (QA drops after first rewrite?)
  4. Answer F1 step0 > step3        (cumulative QA degradation?)

Run for: each instruction_type individually + by group (style vs content).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="15q", help="Dataset tag (e.g. 15q, 200q).")
    return parser.parse_args()


_args  = parse_args()
TAG    = _args.tag
OFS_CSV = REPO_ROOT / "results" / TAG / f"rewriting_chains_{TAG}_openfactscore.csv"
BS_CSV  = REPO_ROOT / "results" / TAG / f"rewriting_chains_{TAG}_bertscore.csv"
F1_CSV  = REPO_ROOT / "results" / TAG / f"rewriting_chains_{TAG}_answer_f1.csv"

INSTRUCTIONS = ["elaborate", "shorten", "paraphrase", "formality"]
GROUPS       = ["content", "style"]


def sign_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided sign test: H0 = P(a > b) = 0.5. Returns result dict."""
    diff = a - b
    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    n   = pos + neg
    if n == 0:
        return {"pos": 0, "neg": 0, "n": 0, "p": float("nan"), "direction": "n/a"}
    result = stats.binomtest(min(pos, neg), n, 0.5, alternative="two-sided")
    direction = "a>b" if pos > neg else "a<b" if neg > pos else "tie"
    return {"pos": pos, "neg": neg, "n": n, "p": result.pvalue, "direction": direction}


def stars(p: float) -> str:
    if np.isnan(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def print_result(label: str, r: dict):
    s = stars(r["p"])
    print(f"  {label}")
    print(f"    n={r['n']}, pos={r['pos']}, neg={r['neg']} | direction={r['direction']} | p={r['p']:.4f} {s}")


def run_tests(label: str, ofs: pd.DataFrame, bs: pd.DataFrame, f1: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # --- 1. FactScore step1 > step3 ---
    if not ofs.empty:
        ofs1 = ofs[ofs["step"] == 1].set_index(["qid", "run"])["factscore"]
        ofs3 = ofs[ofs["step"] == 3].set_index(["qid", "run"])["factscore"]
        idx  = ofs1.index.intersection(ofs3.index)
        r = sign_test(ofs1[idx].values, ofs3[idx].values)
        print_result("FactScore step1 vs step3 (faithfulness degrades?)", r)

    # --- 2. BERTScore baseline step1 > step3 ---
    if not bs.empty:
        bs1 = bs[bs["step"] == 1].set_index(["qid", "run"])["bert_f1_baseline"]
        bs3 = bs[bs["step"] == 3].set_index(["qid", "run"])["bert_f1_baseline"]
        idx  = bs1.index.intersection(bs3.index)
        r = sign_test(bs1[idx].values, bs3[idx].values)
        print_result("BERTScore baseline step1 vs step3 (drift grows?)", r)

    # --- 3. Answer F1 step0 vs step1 ---
    if not f1.empty:
        f1_0 = f1[f1["step"] == 0].set_index(["qid", "run"])["answer_f1"]
        f1_1 = f1[f1["step"] == 1].set_index(["qid", "run"])["answer_f1"]
        idx  = f1_0.index.intersection(f1_1.index)
        r = sign_test(f1_0[idx].values, f1_1[idx].values)
        print_result("Answer F1 step0 vs step1 (QA drops after 1st rewrite?)", r)

    # --- 4. Answer F1 step0 vs step3 ---
    if not f1.empty:
        f1_3 = f1[f1["step"] == 3].set_index(["qid", "run"])["answer_f1"]
        idx  = f1_0.index.intersection(f1_3.index)
        r = sign_test(f1_0[idx].values, f1_3[idx].values)
        print_result("Answer F1 step0 vs step3 (cumulative QA degradation?)", r)


def main():
    ofs = pd.read_csv(OFS_CSV)
    bs  = pd.read_csv(BS_CSV)
    f1  = pd.read_csv(F1_CSV)

    print("\n" + "#"*60)
    print("  SIGN TESTS BY INSTRUCTION TYPE")
    print("#"*60)

    for instr in INSTRUCTIONS:
        run_tests(
            f"instruction_type = {instr.upper()}",
            ofs[ofs["instruction_type"] == instr],
            bs[bs["instruction_type"]  == instr],
            f1[f1["instruction_type"]  == instr],
        )

    print("\n\n" + "#"*60)
    print("  SIGN TESTS BY GROUP (style vs content)")
    print("#"*60)

    for group in GROUPS:
        run_tests(
            f"group = {group.upper()}",
            ofs[ofs["group"] == group],
            bs[bs["group"]   == group],
            f1[f1["group"]   == group],
        )

    print("\n\n" + "#"*60)
    print("  CONCLUSIONS")
    print("#"*60)

    print("""
By instruction type
-------------------
- SHORTEN:    expected to compress; FactScore/BERT degradation signals whether
              compression also removes faithful content.
- PARAPHRASE: should preserve meaning; significant degradation = the model
              is not paraphrasing faithfully.
- FORMALITY:  style-only change; degradation here is surprising and suggests
              content leakage from style edits.
- ELABORATE:  should expand; degradation confirms the model compresses instead.

By group
--------
- CONTENT (shorten + elaborate): both manipulate information quantity.
  Significant degradation expected and interpretable.
- STYLE (formality + paraphrase): should leave facts intact.
  Any significant degradation is a fidelity failure.

Legend: *** p<0.001  ** p<0.01  * p<0.05  ns not significant
        direction a>b = first quantity larger than second
""")


if __name__ == "__main__":
    main()
