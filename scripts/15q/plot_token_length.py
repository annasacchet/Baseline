"""
plot_token_length.py — token length trajectories by instruction type.

Shows mean n_tokens (± 95% CI) across steps 0-3 for each instruction,
for a given dataset tag (default: 15q).

Usage:
  python scripts/15q/plot_token_length.py           # 15q
  python scripts/15q/plot_token_length.py --tag 300q
  python scripts/15q/plot_token_length.py --tag both  # overlay 15q vs 300q
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

INSTRUCTION_ORDER  = ["elaborate", "shorten", "paraphrase", "formality"]
INSTRUCTION_LABELS = {
    "elaborate":  "Elaborate",
    "shorten":    "Shorten",
    "paraphrase": "Paraphrase",
    "formality":  "Formality",
}
COLORS = {
    "elaborate":  "#e15759",
    "shorten":    "#4e79a7",
    "paraphrase": "#f28e2b",
    "formality":  "#59a14f",
}

STEP_LABELS = ["E₀\n(original)", "E₁", "E₂", "E₃"]


def load(tag: str) -> pd.DataFrame:
    path = REPO_ROOT / "results" / tag / f"rewriting_chains_{tag}.csv"
    df = pd.read_csv(path)
    df["tag"] = tag
    return df


def ci95(series):
    n = len(series)
    if n < 2:
        return 0.0
    return 1.96 * series.std() / np.sqrt(n)


def plot_single(df: pd.DataFrame, tag: str, ax: plt.Axes):
    for instr in INSTRUCTION_ORDER:
        sub = df[df["instruction_type"] == instr]
        stats = sub.groupby("step")["n_tokens"].agg(["mean", ci95]).reset_index()
        stats.columns = ["step", "mean", "ci"]
        ax.plot(
            stats["step"], stats["mean"],
            marker="o", linewidth=2, markersize=6,
            color=COLORS[instr], label=INSTRUCTION_LABELS[instr],
        )
        ax.fill_between(
            stats["step"],
            stats["mean"] - stats["ci"],
            stats["mean"] + stats["ci"],
            alpha=0.15, color=COLORS[instr],
        )

    # Horizontal reference = mean E0
    e0_mean = df[df["step"] == 0]["n_tokens"].mean()
    ax.axhline(e0_mean, color="gray", linewidth=1, linestyle="--", alpha=0.6, label="E₀ mean")

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(STEP_LABELS)
    ax.set_xlabel("Rewriting step")
    ax.set_ylabel("Token count (mean ± 95% CI)")
    ax.set_title(f"Token length by instruction — {tag}")
    ax.legend(framealpha=0.9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.25)


def plot_both(df15: pd.DataFrame, df300: pd.DataFrame, axs):
    for ax, df, tag in [(axs[0], df15, "15q"), (axs[1], df300, "300q")]:
        plot_single(df, tag, ax)
    # shared y-axis
    ymin = min(ax.get_ylim()[0] for ax in axs)
    ymax = max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(ymin, ymax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="15q", help="Dataset tag: 15q, 300q, or 'both'")
    args = parser.parse_args()

    plot_dir = REPO_ROOT / "results" / "plots"

    if args.tag == "both":
        df15  = load("15q")
        df300 = load("300q")
        fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
        plot_both(df15, df300, axs)
        fig.suptitle("Token length trajectories: 15q vs 300q", fontsize=13, y=1.01)
        fig.tight_layout()
        out_pdf = plot_dir / "token_length_both.pdf"
        out_png = plot_dir / "png" / "token_length_both.png"
    else:
        df = load(args.tag)
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_single(df, args.tag, ax)
        fig.tight_layout()
        out_pdf = plot_dir / args.tag / f"token_length_{args.tag}.pdf"
        out_png = plot_dir / args.tag / "png" / f"token_length_{args.tag}.png"

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
