"""
One plot, one question: does rewriting degrade the ability to answer?

Two panels side by side:
  Left  — all 180 chains (most start at F1=0, signal is buried)
  Right — only chains answerable at step 1 (F1>0), where degradation is visible

Each panel: mean line per instruction type + shaded 1-std band.
No individual chain lines — too noisy. Clean and readable.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
F1_CSV  = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q_answer_f1.csv"
OUT_DIR = REPO_ROOT / "results" / "plots" / "15q"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "formality":  "#4e79a7",
    "paraphrase": "#f28e2b",
    "shorten":    "#e15759",
    "elaborate":  "#59a14f",
}
INSTRUCTIONS = ["formality", "paraphrase", "shorten", "elaborate"]
STEPS = [1, 2, 3]


def load():
    df = pd.read_csv(F1_CSV)
    df["chain_id"] = df["qid"] + "|" + df["instruction_type"] + "|" + df["run"].astype(str)
    return df


def draw_panel(ax, df, title, note):
    for instr in INSTRUCTIONS:
        sub = df[df["instruction_type"] == instr]
        means = sub.groupby("step")["answer_f1"].mean().reindex(STEPS)
        stds  = sub.groupby("step")["answer_f1"].std().reindex(STEPS)
        c = COLORS[instr]
        ax.plot(STEPS, means.values, color=c, linewidth=2.5,
                marker="o", markersize=7, markerfacecolor="white",
                markeredgewidth=2, label=instr, zorder=3)
        ax.fill_between(STEPS,
                         (means - stds).clip(0).values,
                         (means + stds).clip(upper=1).values,
                         color=c, alpha=0.10, zorder=1)
        # label endpoint
        ax.text(3.07, float(means.iloc[-1]),
                f"{means.iloc[-1]:.2f}",
                va="center", fontsize=8.5, color=c, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Rewriting step", fontsize=10)
    ax.set_ylabel("Answer F1", fontsize=10)
    ax.set_xticks(STEPS)
    ax.set_xticklabels(["Step 1\n(first rewrite)", "Step 2", "Step 3\n(third rewrite)"],
                       fontsize=9)
    ax.set_xlim(0.7, 3.5)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.text(0.02, 0.97, note, transform=ax.transAxes,
            fontsize=8, va="top", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"))


def main():
    df = load()

    # Panel right: only chains where F1 > 0 at step 1
    answerable_chains = (
        df[df["step"] == 1]
        .query("answer_f1 > 0")["chain_id"]
        .unique()
    )
    df_ans = df[df["chain_id"].isin(answerable_chains) & (df["step"] >= 1)]
    df_all = df[df["step"] >= 1]

    n_all = df_all["chain_id"].nunique()
    n_ans = df_ans["chain_id"].nunique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    draw_panel(ax1, df_all,
               f"All chains  (n={n_all})",
               "Includes chains where QA model\ncannot answer even from E₀")

    draw_panel(ax2, df_ans,
               f"Answerable chains only  (n={n_ans})",
               "Only chains where F1 > 0\nat step 1 — degradation visible here")

    # shared legend
    handles = [mpatches.Patch(color=COLORS[i], label=i) for i in INSTRUCTIONS]
    fig.legend(handles=handles, title="Instruction type",
               loc="lower center", ncol=4, fontsize=9,
               title_fontsize=9, bbox_to_anchor=(0.5, -0.04),
               framealpha=0.9)

    fig.suptitle("Does rewriting degrade the ability to answer questions?",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, 0.97,
             "Mean Answer F1 across steps 1→3  (band = ±1 std)",
             ha="center", fontsize=9, color="#555")

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = OUT_DIR / "degradation_f1.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
