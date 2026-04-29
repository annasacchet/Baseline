"""
Visualizzazione token counts per il pilota 15q (5 domande × 3 hop).

Aggrega le 45 chain per (group, instruction_type, step) — non si possono
plottare tutte sovrapposte, illeggibile. Uso mediana + banda IQR (25/75 percentile)
perché le distribuzioni di n_tokens sono molto asimmetriche con coda lunga
(min 194, max 3276).

Output:
  - tokens_15q_by_instruction.pdf  — un pannello per instruction_type (4)
  - tokens_15q_summary.pdf         — overlay di tutte e 4 le istruzioni
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "results" / "rewriting_chains_15q.csv"
OUT_BY_INSTR = REPO_ROOT / "results" / "tokens_15q_by_instruction.pdf"
OUT_SUMMARY = REPO_ROOT / "results" / "tokens_15q_summary.pdf"

COLORS = {
    "elaborate": "#e74c3c",
    "shorten": "#3498db",
    "formality": "#2ecc71",
    "paraphrase": "#f39c12",
}
MARKERS = {
    "elaborate": "o",
    "shorten": "s",
    "formality": "^",
    "paraphrase": "D",
}
GROUP_OF = {
    "elaborate": "content",
    "shorten": "content",
    "formality": "style",
    "paraphrase": "style",
}


def load_and_summarize(csv_path: Path) -> pd.DataFrame:
    """Aggrega n_tokens per (instruction_type, step). Returns one row per cell."""
    df = pd.read_csv(csv_path)
    summary = (
        df.groupby(["instruction_type", "step"])["n_tokens"]
        .agg(
            median="median",
            mean="mean",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            n="count",
        )
        .reset_index()
    )
    return df, summary


def plot_by_instruction(summary: pd.DataFrame, e0_median: float, out_path: Path):
    """One panel per instruction_type. Median line + IQR band."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, itype in enumerate(["elaborate", "shorten", "formality", "paraphrase"]):
        ax = axes[i]
        sub = summary[summary["instruction_type"] == itype].sort_values("step")

        ax.fill_between(
            sub["step"], sub["q25"], sub["q75"],
            color=COLORS[itype], alpha=0.25, label="IQR (25–75 percentile)",
        )
        ax.plot(
            sub["step"], sub["median"],
            marker=MARKERS[itype], markersize=10, linewidth=2.5,
            color=COLORS[itype], label="Median",
        )

        ax.axhline(e0_median, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(
            0.02, e0_median, f"E₀ median = {e0_median:.0f}",
            color="gray", fontsize=9, va="bottom",
            transform=ax.get_yaxis_transform(),
        )

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("Step")
        ax.set_ylabel("Token count (OLMo tokenizer)")
        ax.set_title(f"{itype.capitalize()} ({GROUP_OF[itype]})")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Token count per step — pilot 15q (5 questions × 3 hop)\n"
        "Median across 45 chains per cell · IQR band",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def plot_summary(summary: pd.DataFrame, e0_median: float, out_path: Path):
    """All 4 instructions overlaid — one figure."""
    fig, ax = plt.subplots(figsize=(11, 7))

    for itype in ["elaborate", "shorten", "formality", "paraphrase"]:
        sub = summary[summary["instruction_type"] == itype].sort_values("step")
        ax.plot(
            sub["step"], sub["median"],
            marker=MARKERS[itype], markersize=10, linewidth=2.5,
            color=COLORS[itype],
            label=f"{itype} ({GROUP_OF[itype]})",
        )

    ax.axhline(e0_median, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(
        3.05, e0_median, f"E₀ median\n{e0_median:.0f} tok",
        color="gray", fontsize=9, va="center",
    )

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Token count (OLMo tokenizer, median)", fontsize=12)
    ax.set_title(
        "Length collapse across rewriting steps — pilot 15q\n"
        "Median across 45 chains per (instruction, step)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def main():
    df, summary = load_and_summarize(CSV_PATH)

    e0_median = df[df["step"] == 0]["n_tokens"].median()

    print("=" * 70)
    print("TOKEN COUNTS — pilot 15q (5 questions × 3 hop)")
    print("=" * 70)
    print(f"Total rows: {len(df)}")
    print(f"Chains: {df.groupby(['qid','group','instruction_type','run']).ngroups}")
    print(f"E₀ token count — median: {e0_median:.0f}")
    print()
    print("Median tokens per (instruction, step) — IQR in [q25, q75]:")
    print("-" * 70)
    pivot = summary.pivot(index="instruction_type", columns="step", values="median")
    print(pivot.round(0).astype(int))
    print()
    print("Drop from E₀ to step 1 (median):")
    for itype in ["elaborate", "shorten", "formality", "paraphrase"]:
        m1 = summary[(summary["instruction_type"] == itype) & (summary["step"] == 1)]["median"].iloc[0]
        drop = (e0_median - m1) / e0_median * 100
        print(f"  {itype:12s}: {e0_median:.0f} → {m1:.0f}  ({drop:+.1f}%)")
    print()

    plot_by_instruction(summary, e0_median, OUT_BY_INSTR)
    plot_summary(summary, e0_median, OUT_SUMMARY)


if __name__ == "__main__":
    main()
