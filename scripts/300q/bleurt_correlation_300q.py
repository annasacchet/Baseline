"""
Correlation analysis: BLEURT vs Answer F1 and BLEURT vs BERTScore.

Two comparisons:
  1. bleurt_answer vs answer_f1         — do they agree on answer quality degradation?
  2. bleurt_baseline vs bert_f1_baseline — do they agree on text drift from E_0?

Output:
  - Pearson + Spearman correlations (overall, per instruction_type, per step)
  - Scatter plots saved to results/300q/

Uso:
  python3 scripts/300q/bleurt_correlation_300q.py
  python3 scripts/300q/bleurt_correlation_300q.py --save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "300q"
PLOTS_DIR = REPO_ROOT / "results" / "plots" / "300q"

FILES = {
    "bleurt": RESULTS_DIR / "rewriting_chains_300q_bleurt.csv",
    "f1": RESULTS_DIR / "rewriting_chains_300q_answer_f1_olmo31_4bit.csv",
    "bertscore": RESULTS_DIR / "rewriting_chains_300q_bertscore.csv",
}

CHAIN_KEYS = ["qid", "group", "instruction_type", "run", "step"]


def corr_row(x: pd.Series, y: pd.Series) -> dict:
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return {"n": len(x), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {"n": len(x), "pearson_r": round(pr, 3), "pearson_p": round(pp, 4),
            "spearman_r": round(sr, 3), "spearman_p": round(sp, 4)}


def print_corr_table(df: pd.DataFrame, x_col: str, y_col: str, groupby: list[str], title: str) -> pd.DataFrame:
    print(f"\n{title}")
    print("-" * len(title))
    rows = []
    for keys, grp in df.groupby(groupby):
        row = corr_row(grp[x_col], grp[y_col])
        if isinstance(keys, tuple):
            row = dict(zip(groupby, keys)) | row
        else:
            row = {groupby[0]: keys} | row
        rows.append(row)
    overall = corr_row(df[x_col], df[y_col])
    overall = {"overall": True} | overall
    rows.append(overall)
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    return result


def scatter(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str,
            xlabel: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"elaborate": "#4C72B0", "formality": "#DD8452",
              "paraphrase": "#55A868", "shorten": "#C44E52"}
    for itype, grp in df.groupby(hue_col):
        ax.scatter(grp[x_col], grp[y_col], alpha=0.25, s=12,
                   color=colors.get(itype, "gray"), label=itype)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=hue_col, fontsize=8, markerscale=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save correlation CSVs to results/300q/stats/")
    args = parser.parse_args()

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} CSV not found: {path}")

    bleurt = pd.read_csv(FILES["bleurt"])
    f1 = pd.read_csv(FILES["f1"])[CHAIN_KEYS + ["answer_f1"]]
    bert = pd.read_csv(FILES["bertscore"])[CHAIN_KEYS + ["bert_f1_baseline", "bert_f1_consecutive"]]

    # Join all on chain keys
    df = bleurt.merge(f1, on=CHAIN_KEYS, how="inner")
    df = df.merge(bert, on=CHAIN_KEYS, how="inner")

    print(f"Joined rows: {len(df)}  (steps > 0 only in bleurt/bertscore)")

    # -------------------------------------------------------------------------
    # 1. BLEURT answer vs Answer F1
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BLEURT answer  vs  Answer F1")
    print("=" * 60)

    c1_instr = print_corr_table(df, "bleurt_answer", "answer_f1",
                                ["instruction_type"],
                                "Per instruction_type")
    c1_step = print_corr_table(df, "bleurt_answer", "answer_f1",
                               ["step"],
                               "Per step")
    c1_both = print_corr_table(df, "bleurt_answer", "answer_f1",
                               ["instruction_type", "step"],
                               "Per instruction_type x step")

    scatter(df, "bleurt_answer", "answer_f1", "instruction_type",
            "BLEURT answer", "Answer F1",
            "BLEURT answer vs Answer F1",
            PLOTS_DIR / "png" / "bleurt_vs_f1_scatter.png")
    scatter(df, "bleurt_answer", "answer_f1", "instruction_type",
            "BLEURT answer", "Answer F1",
            "BLEURT answer vs Answer F1",
            PLOTS_DIR / "bleurt_vs_f1_scatter.pdf")

    # -------------------------------------------------------------------------
    # 2. BLEURT baseline vs BERTScore baseline
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BLEURT baseline  vs  BERTScore F1 baseline")
    print("=" * 60)

    c2_instr = print_corr_table(df, "bleurt_baseline", "bert_f1_baseline",
                                ["instruction_type"],
                                "Per instruction_type")
    c2_step = print_corr_table(df, "bleurt_baseline", "bert_f1_baseline",
                               ["step"],
                               "Per step")

    scatter(df, "bleurt_baseline", "bert_f1_baseline", "instruction_type",
            "BLEURT baseline", "BERTScore F1 baseline",
            "BLEURT baseline vs BERTScore baseline",
            PLOTS_DIR / "png" / "bleurt_vs_bertscore_scatter.png")
    scatter(df, "bleurt_baseline", "bert_f1_baseline", "instruction_type",
            "BLEURT baseline", "BERTScore F1 baseline",
            "BLEURT baseline vs BERTScore baseline",
            PLOTS_DIR / "bleurt_vs_bertscore_scatter.pdf")

    # -------------------------------------------------------------------------
    # 3. BLEURT consecutive vs BERTScore consecutive
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BLEURT consecutive  vs  BERTScore F1 consecutive")
    print("=" * 60)

    c3_instr = print_corr_table(df, "bleurt_consecutive", "bert_f1_consecutive",
                                ["instruction_type"],
                                "Per instruction_type")

    scatter(df, "bleurt_consecutive", "bert_f1_consecutive", "instruction_type",
            "BLEURT consecutive", "BERTScore F1 consecutive",
            "BLEURT consecutive vs BERTScore consecutive",
            PLOTS_DIR / "png" / "bleurt_vs_bertscore_consecutive_scatter.png")
    scatter(df, "bleurt_consecutive", "bert_f1_consecutive", "instruction_type",
            "BLEURT consecutive", "BERTScore F1 consecutive",
            "BLEURT consecutive vs BERTScore consecutive",
            PLOTS_DIR / "bleurt_vs_bertscore_consecutive_scatter.pdf")

    if args.save:
        stats_dir = RESULTS_DIR / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        c1_instr.to_csv(stats_dir / "bleurt_vs_f1_by_instruction.csv", index=False)
        c1_step.to_csv(stats_dir / "bleurt_vs_f1_by_step.csv", index=False)
        c1_both.to_csv(stats_dir / "bleurt_vs_f1_by_instruction_step.csv", index=False)
        c2_instr.to_csv(stats_dir / "bleurt_vs_bertscore_by_instruction.csv", index=False)
        c2_step.to_csv(stats_dir / "bleurt_vs_bertscore_by_step.csv", index=False)
        c3_instr.to_csv(stats_dir / "bleurt_vs_bertscore_consecutive_by_instruction.csv", index=False)
        print(f"\n[saved] {stats_dir}/bleurt_vs_*.csv")


if __name__ == "__main__":
    main()
