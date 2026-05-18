"""OFS plots for 300q: lineplots (rescaled) + heatmap per qid × step by hop."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV  = REPO / "results/300q/rewriting_chains_300q_openfactscore.csv"
OUT_DIR = REPO / "results/300q"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTR_ORDER  = ["formality", "paraphrase", "shorten", "elaborate"]
INSTR_COLORS = {
    "formality": "#1f77b4",
    "paraphrase": "#2ca02c",
    "shorten":    "#d62728",
    "elaborate":  "#9467bd",
}
HOP_ORDER  = [2, 3, 4]
HOP_COLORS = {2: "#e15759", 3: "#f28e2b", 4: "#4e79a7"}
STEPS = [1, 2, 3]
YLIM  = (0.80, 1.0)


def hop_count(qid: str) -> int:
    m = re.match(r"(\d+)hop", qid)
    return int(m.group(1)) if m else 0


def agg_by(df: pd.DataFrame, groupcol: str) -> pd.DataFrame:
    agg = (
        df.groupby([groupcol, "step"])["factscore"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / agg["count"].pow(0.5)
    return agg


def lineplot(agg, groupcol, order, colors, title, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key in order:
        sub = agg[agg[groupcol] == key].sort_values("step")
        if sub.empty:
            continue
        ax.errorbar(
            sub["step"], sub["mean"], yerr=sub["sem"],
            marker="o", linewidth=2, capsize=3,
            label=str(key), color=colors[key],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['mean']:.3f}",
                xy=(row["step"], row["mean"]),
                xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=7.5, color=colors[key],
            )
    ax.set_xlabel("Rewriting step")
    ax.set_ylabel("OpenFactScore (mean ± SEM)")
    ax.set_title(title)
    ax.set_xticks(STEPS)
    ax.set_ylim(*YLIM)
    ax.grid(True, alpha=0.3)
    ax.legend(title=groupcol, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")
    plt.close(fig)


def heatmap(df: pd.DataFrame, out_path: Path) -> None:
    pivot_full = (
        df.groupby(["qid", "step"])["factscore"]
        .mean().unstack("step").reindex(columns=STEPS)
    )
    pivot_full["hop"] = pivot_full.index.map(hop_count)

    panels = []
    for h in HOP_ORDER:
        sub = pivot_full[pivot_full["hop"] == h].drop(columns="hop")
        sub = sub.assign(mean=sub.mean(axis=1)).sort_values("mean", ascending=False).drop(columns="mean")
        panels.append((h, sub))

    ratios = []
    for _ in panels:
        ratios += [0.05, 1.0, 0.05]

    fig = plt.figure(figsize=(11, 8.5))
    gs  = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.02)

    im = None
    for i, (h, pivot) in enumerate(panels):
        ax_strip = fig.add_subplot(gs[0, i * 3])
        ax       = fig.add_subplot(gs[0, i * 3 + 1])

        n = len(pivot)
        ax_strip.imshow(
            np.full((n, 1), h, dtype=float), aspect="auto",
            cmap=plt.matplotlib.colors.ListedColormap([HOP_COLORS[h]]),
            vmin=h - 0.5, vmax=h + 0.5, interpolation="nearest",
        )
        ax_strip.set_xticks([])
        ax_strip.set_yticks([])
        ax_strip.spines[:].set_visible(False)
        r, g, b = plt.matplotlib.colors.to_rgb(HOP_COLORS[h])
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        ax_strip.text(
            0, n / 2, f"{h}-hop  (n={n})",
            ha="center", va="center", fontsize=10,
            color="white" if lum < 0.55 else "black",
            fontweight="bold", rotation=90,
        )

        im = ax.imshow(
            pivot.values, aspect="auto", cmap="RdYlGn",
            vmin=YLIM[0], vmax=YLIM[1], interpolation="nearest",
        )
        ax.set_xticks(range(len(STEPS)))
        ax.set_xticklabels([f"Step {s}" for s in STEPS], fontsize=9)
        ax.set_yticks([])

    fig.suptitle(
        "OpenFactScore per question × step — 300q\n"
        "(mean across instructions and runs; questions sorted by mean OFS, desc)",
        fontsize=12, fontweight="bold",
    )
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="OpenFactScore")
    fig.subplots_adjust(left=0.04, right=0.91, top=0.90, bottom=0.06)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    print(f"Saved: {out_path}")
    plt.close(fig)


df = pd.read_csv(CSV)
df["n_hop"] = df["qid"].apply(hop_count)

lineplot(
    agg_by(df, "instruction_type"), "instruction_type",
    INSTR_ORDER, INSTR_COLORS,
    "OpenFactScore across rewriting steps — 300q (by instruction type)",
    OUT_DIR / "ofs_by_instruction_300q.png",
)

lineplot(
    agg_by(df, "n_hop"), "n_hop",
    HOP_ORDER, HOP_COLORS,
    "OpenFactScore across rewriting steps — 300q (by n_hop)",
    OUT_DIR / "ofs_by_hop_300q.png",
)

heatmap(df, OUT_DIR / "ofs_heatmap_300q.png")
