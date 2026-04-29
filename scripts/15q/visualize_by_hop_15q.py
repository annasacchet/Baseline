"""
Per-hop decomposition of the 15q pilot results.

The pilot has 5 questions per hop (2hop, 3hop, 4hop). Hop count is encoded in
the qid prefix ("2hop__...", "3hop__...", "4hop__..."). MuSiQue 4hop questions
have longer evidence and require more reasoning steps, so it is plausible that
the rewriting model behaves differently across hop counts.

Outputs:
  - by_hop_15q_length.pdf      n_tokens median per (instruction, step, hop)
  - by_hop_15q_bertscore.pdf   bert_f1_baseline median per (instruction, step, hop)
  - by_hop_15q_summary.pdf     overlay: same axes as combined plot, but split by hop
  - prints a comparison table (hop x metric x step x instruction)
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAINS_CSV = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q.csv"
BERT_CSV = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q_bertscore.csv"

OUT_LENGTH = REPO_ROOT / "results" / "plots" / "15q" / "by_hop_15q_length.pdf"
OUT_BERT = REPO_ROOT / "results" / "plots" / "15q" / "by_hop_15q_bertscore.pdf"
OUT_SUMMARY = REPO_ROOT / "results" / "plots" / "15q" / "by_hop_15q_summary.pdf"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]
INSTRUCTIONS = ["elaborate", "shorten", "formality", "paraphrase"]
HOPS = [2, 3, 4]

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
HOP_LINESTYLE = {
    2: "-",
    3: "--",
    4: ":",
}
HOP_ALPHA = {
    2: 1.0,
    3: 0.85,
    4: 0.7,
}
GROUP_OF = {
    "elaborate": "content",
    "shorten": "content",
    "formality": "style",
    "paraphrase": "style",
}


def hop_count_from_qid(qid: str) -> int:
    # MuSiQue qids encode hop count as a number prefix followed by "hop":
    # "2hop__...", "3hop1__...", "3hop2__...", "4hop1__...", "4hop3__..."
    m = re.match(r"(\d+)hop", qid)
    return int(m.group(1)) if m else -1


def load_with_hop() -> pd.DataFrame:
    chains = pd.read_csv(CHAINS_CSV)
    bert = pd.read_csv(BERT_CSV)
    df = chains.merge(bert, on=CHAIN_KEYS + ["step"], how="left")
    df["hop"] = df["qid"].apply(hop_count_from_qid)
    return df


def aggregate(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.groupby(["instruction_type", "step", "hop"])[value_col]
        .median()
        .reset_index(name="median")
    )


def plot_metric_by_hop(df: pd.DataFrame, value_col: str, ylabel: str, title: str, out_path: Path, ylim=None, step_filter=None):
    """4 panels (one per instruction). Inside each: 3 lines (one per hop)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    medians = aggregate(df, value_col)
    if step_filter is not None:
        medians = medians[medians["step"].isin(step_filter)]

    for i, itype in enumerate(INSTRUCTIONS):
        ax = axes[i]
        for hop in HOPS:
            sub = medians[(medians["instruction_type"] == itype) & (medians["hop"] == hop)].sort_values("step")
            if sub.empty:
                continue
            ax.plot(
                sub["step"], sub["median"],
                marker=MARKERS[itype], markersize=9, linewidth=2.2,
                color=COLORS[itype],
                linestyle=HOP_LINESTYLE[hop],
                alpha=HOP_ALPHA[hop],
                label=f"{hop}-hop",
            )

        ax.set_xticks(sorted(medians["step"].unique()))
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{itype.capitalize()} ({GROUP_OF[itype]})")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=10, title="hop count")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def plot_combined_by_hop(df: pd.DataFrame, out_path: Path):
    """3 panels (one per hop). Inside each: 4 instruction trajectories in (length, BERTScore F1) space."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    medians_len = aggregate(df, "n_tokens")
    medians_f1 = aggregate(df, "bert_f1_baseline")

    e0_by_hop = df[df["step"] == 0].groupby("hop")["n_tokens"].median().to_dict()

    for i, hop in enumerate(HOPS):
        ax = axes[i]
        for itype in INSTRUCTIONS:
            len_sub = medians_len[(medians_len["instruction_type"] == itype) & (medians_len["hop"] == hop) & (medians_len["step"] > 0)].sort_values("step")
            f1_sub = medians_f1[(medians_f1["instruction_type"] == itype) & (medians_f1["hop"] == hop) & (medians_f1["step"] > 0)].sort_values("step")

            xs = len_sub["median"].values
            ys = f1_sub["median"].values
            steps = len_sub["step"].values

            for k in range(len(xs) - 1):
                ax.annotate(
                    "",
                    xy=(xs[k + 1], ys[k + 1]),
                    xytext=(xs[k], ys[k]),
                    arrowprops=dict(arrowstyle="->", color=COLORS[itype], lw=2.0, alpha=0.7),
                )
            ax.scatter(
                xs, ys,
                s=200, color=COLORS[itype], marker=MARKERS[itype],
                edgecolors="black", linewidth=1.2, zorder=5,
                label=f"{itype} ({GROUP_OF[itype]})" if i == 0 else None,
            )
            for x, y, s in zip(xs, ys, steps):
                ax.annotate(
                    f"t{int(s)}",
                    (x, y),
                    xytext=(7, 7),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    color=COLORS[itype],
                )

        e0 = e0_by_hop.get(hop)
        if e0:
            ax.axvline(e0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.text(e0, 0.795, f"E₀={e0:.0f}", color="gray", fontsize=9, ha="right")

        ax.set_xlabel("Length — median n_tokens", fontsize=11)
        ax.set_ylabel("BERTScore F1 (Baseline)", fontsize=11)
        ax.set_title(f"{hop}-hop questions (n=5)", fontsize=12, fontweight="bold")
        ax.set_ylim(0.79, 0.95)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        "Degradation trajectories split by hop count — pilot 15q\n"
        "arrows: t₁ → t₂ → t₃ · medians across 9 chains per cell (5 qid × 3 wording — wait, actually 15 chains: 5 qid × 3 wording)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def main():
    df = load_with_hop()
    df_with_bert = df[df["bert_f1_baseline"].notna() | (df["step"] == 0)].copy()

    print("=" * 70)
    print("Per-hop decomposition — pilot 15q")
    print("=" * 70)

    counts = df.groupby("hop")["qid"].nunique()
    print(f"\nQuestions per hop: {counts.to_dict()}")

    rows_per_cell = df.groupby(["instruction_type", "step", "hop"]).size()
    print(f"Rows per (instruction, step, hop): expected 15  ·  observed range: {rows_per_cell.min()}–{rows_per_cell.max()}")
    print()

    print("Median n_tokens per (instruction, hop, step):")
    print("-" * 70)
    pivot_len = (
        aggregate(df, "n_tokens")
        .pivot_table(index=["instruction_type", "hop"], columns="step", values="median")
    )
    print(pivot_len.round(0).astype("Int64"))
    print()

    print("Median BERTScore F1 (Baseline) per (instruction, hop, step):")
    print("-" * 70)
    pivot_f1 = (
        aggregate(df_with_bert, "bert_f1_baseline")
        .pivot_table(index=["instruction_type", "hop"], columns="step", values="median")
    )
    print(pivot_f1.round(3))
    print()

    plot_metric_by_hop(
        df, "n_tokens",
        ylabel="Token count (median)",
        title="Length per step, split by hop count — pilot 15q\n(median across 15 chains per cell: 5 qid × 3 wording)",
        out_path=OUT_LENGTH,
    )
    plot_metric_by_hop(
        df_with_bert, "bert_f1_baseline",
        ylabel="BERTScore F1 (Baseline)",
        title="BERTScore Baseline per step, split by hop count — pilot 15q\n(median across 15 chains per cell)",
        out_path=OUT_BERT,
        ylim=(0.79, 0.95),
        step_filter=[1, 2, 3],
    )
    plot_combined_by_hop(df_with_bert, OUT_SUMMARY)


if __name__ == "__main__":
    main()
