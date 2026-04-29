"""
Combined visualization: Length × BERTScore Baseline (15q pilot).

Each instruction is a trajectory in (n_tokens, BERTScore F1 baseline) space:
  - markers at t1, t2, t3 (medians across 45 chains per cell)
  - arrows show direction t1 → t2 → t3
  - E_0 reference (median length, F1=1.0 by definition) marks the starting point

Reads:
  - results/rewriting_chains_15q.csv             (n_tokens)
  - results/rewriting_chains_15q_bertscore.csv   (bert_f1_baseline)

Joins on (qid, group, instruction_type, run, step), aggregates median per
(instruction_type, step), and plots one trajectory per instruction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAINS_CSV = REPO_ROOT / "results" / "rewriting_chains_15q.csv"
BERT_CSV = REPO_ROOT / "results" / "rewriting_chains_15q_bertscore.csv"
OUT_PATH = REPO_ROOT / "results" / "length_x_bertscore_15q.pdf"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

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


def load_joined() -> pd.DataFrame:
    """Inner-join chain CSV with BERTScore CSV on chain keys + step."""
    chains = pd.read_csv(CHAINS_CSV)
    bert = pd.read_csv(BERT_CSV)
    merged = chains.merge(bert, on=CHAIN_KEYS + ["step"], how="left")
    return merged


def aggregate_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Median of n_tokens and bert_f1_baseline per (instruction_type, step)."""
    return (
        df.groupby(["instruction_type", "step"])
        .agg(
            n_tokens_median=("n_tokens", "median"),
            f1_median=("bert_f1_baseline", "median"),
        )
        .reset_index()
    )


def main():
    print("=" * 70)
    print("Length × BERTScore Baseline — pilot 15q")
    print("=" * 70)

    df = load_joined()
    print(f"Joined rows: {len(df)}")

    medians = aggregate_medians(df)

    # E_0 median length (used as anchor on the right side of the plot)
    e0_median_len = df[df["step"] == 0]["n_tokens"].median()
    print(f"E_0 median length: {e0_median_len:.0f} tokens")
    print()

    print("Trajectories — (n_tokens median, F1 median) per (instruction, step):")
    print("-" * 70)
    print(medians.pivot(index="instruction_type", columns="step", values="n_tokens_median").round(0).astype("Int64"))
    print()
    print(medians.pivot(index="instruction_type", columns="step", values="f1_median").round(3))
    print()

    fig, ax = plt.subplots(figsize=(11, 8))

    for itype in ["elaborate", "shorten", "formality", "paraphrase"]:
        sub = medians[medians["instruction_type"] == itype].sort_values("step")
        sub = sub[sub["step"] > 0]  # F1 baseline is undefined at step 0

        xs = sub["n_tokens_median"].values
        ys = sub["f1_median"].values
        steps = sub["step"].values

        # arrows t1 -> t2 -> t3
        for i in range(len(xs) - 1):
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", color=COLORS[itype], lw=2.2, alpha=0.75),
            )

        # markers
        ax.scatter(
            xs, ys,
            s=260, color=COLORS[itype], marker=MARKERS[itype],
            edgecolors="black", linewidth=1.3, zorder=5,
            label=f"{itype} ({GROUP_OF[itype]})",
        )

        # step labels next to markers
        for x, y, s in zip(xs, ys, steps):
            ax.annotate(
                f"t{int(s)}",
                (x, y),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=COLORS[itype],
            )

    # E_0 reference: median length, F1=1.0 per definizione (sim(E_0, E_0)=1)
    ax.axvline(e0_median_len, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(
        e0_median_len, 1.001, f"E₀ median length = {e0_median_len:.0f} tok",
        color="gray", fontsize=9, ha="right", va="bottom",
        rotation=0,
    )

    ax.set_xlabel("Length — median n_tokens across 45 chains", fontsize=12)
    ax.set_ylabel("BERTScore F1 (Baseline, vs E₀) — median across 45 chains", fontsize=12)
    ax.set_title(
        "Degradation trajectory in (Length, BERTScore) space — pilot 15q\n"
        "arrows go from t₁ → t₂ → t₃",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
