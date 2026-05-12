"""Plot BERTScore Baseline and Consecutive across rewriting steps."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "results/300q/rewriting_chains_300q_bertscore.csv"
OUT_BASELINE_PDF = REPO / "results/plots/300q/bertscore_baseline_by_step.pdf"
OUT_BASELINE_PNG = REPO / "results/plots/300q/png/bertscore_baseline_by_step.png"
OUT_CONSECUTIVE_PDF = REPO / "results/plots/300q/bertscore_consecutive_by_step.pdf"
OUT_CONSECUTIVE_PNG = REPO / "results/plots/300q/png/bertscore_consecutive_by_step.png"
OUT_BOTH_PDF = REPO / "results/plots/300q/bertscore_baseline_consecutive_by_step.pdf"
OUT_BOTH_PNG = REPO / "results/plots/300q/png/bertscore_baseline_consecutive_by_step.png"

INSTR_ORDER = ["formality", "paraphrase", "shorten", "elaborate"]
COLORS = {
    "formality": "#1f77b4",
    "paraphrase": "#2ca02c",
    "shorten": "#d62728",
    "elaborate": "#9467bd",
}

df = pd.read_csv(CSV)

def agg_metric(metric_col):
    return (
        df.groupby(["group", "instruction_type", "step"])[metric_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

agg_baseline = agg_metric("bert_f1_baseline")
agg_baseline["sem"] = agg_baseline["std"] / agg_baseline["count"].pow(0.5)

agg_consecutive = agg_metric("bert_f1_consecutive")
agg_consecutive["sem"] = agg_consecutive["std"] / agg_consecutive["count"].pow(0.5)

# ===== Plot 1: Baseline only =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, group in zip(axes, ["style", "content"]):
    for instr in INSTR_ORDER:
        sub = agg_baseline[(agg_baseline["group"] == group) & (agg_baseline["instruction_type"] == instr)].sort_values("step")
        if sub.empty:
            continue
        ax.errorbar(
            sub["step"], sub["mean"], yerr=sub["sem"],
            marker="o", linewidth=2, capsize=3,
            label=instr, color=COLORS[instr],
        )
    ax.set_xlabel("Rewriting step")
    ax.set_title(f"group = {group}")
    ax.set_xticks([1, 2, 3])
    ax.set_ylim([0.8, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

axes[0].set_ylabel("BERTScore F1 Baseline (mean ± SEM)")
fig.suptitle("BERTScore Baseline (vs original) across rewriting steps — 300q")
fig.tight_layout()

OUT_BASELINE_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_BASELINE_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_BASELINE_PDF)
fig.savefig(OUT_BASELINE_PNG, dpi=160)
print(f"Saved: {OUT_BASELINE_PDF}")

# ===== Plot 2: Consecutive only =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, group in zip(axes, ["style", "content"]):
    for instr in INSTR_ORDER:
        sub = agg_consecutive[(agg_consecutive["group"] == group) & (agg_consecutive["instruction_type"] == instr)].sort_values("step")
        if sub.empty:
            continue
        ax.errorbar(
            sub["step"], sub["mean"], yerr=sub["sem"],
            marker="o", linewidth=2, capsize=3,
            label=instr, color=COLORS[instr],
        )
    ax.set_xlabel("Rewriting step")
    ax.set_title(f"group = {group}")
    ax.set_xticks([1, 2, 3])
    ax.set_ylim([0.8, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

axes[0].set_ylabel("BERTScore F1 Consecutive (mean ± SEM)")
fig.suptitle("BERTScore Consecutive (vs previous step) across rewriting steps — 300q")
fig.tight_layout()

fig.savefig(OUT_CONSECUTIVE_PDF)
fig.savefig(OUT_CONSECUTIVE_PNG, dpi=160)
print(f"Saved: {OUT_CONSECUTIVE_PDF}")

# ===== Plot 3: Both side-by-side (top=baseline, bottom=consecutive) =====
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top row: Baseline
for ax, group in zip(axes[0, :], ["style", "content"]):
    for instr in INSTR_ORDER:
        sub = agg_baseline[(agg_baseline["group"] == group) & (agg_baseline["instruction_type"] == instr)].sort_values("step")
        if sub.empty:
            continue
        ax.errorbar(
            sub["step"], sub["mean"], yerr=sub["sem"],
            marker="o", linewidth=2, capsize=3,
            label=instr, color=COLORS[instr],
        )
    ax.set_title(f"Baseline — group = {group}")
    ax.set_xticks([1, 2, 3])
    ax.set_ylim([0.8, 1.0])
    ax.grid(True, alpha=0.3)
    if group == "style":
        ax.set_ylabel("BERTScore F1")
        ax.legend(frameon=False, title="instruction_type")

# Bottom row: Consecutive
for ax, group in zip(axes[1, :], ["style", "content"]):
    for instr in INSTR_ORDER:
        sub = agg_consecutive[(agg_consecutive["group"] == group) & (agg_consecutive["instruction_type"] == instr)].sort_values("step")
        if sub.empty:
            continue
        ax.errorbar(
            sub["step"], sub["mean"], yerr=sub["sem"],
            marker="o", linewidth=2, capsize=3,
            label=instr, color=COLORS[instr],
        )
    ax.set_xlabel("Rewriting step")
    ax.set_title(f"Consecutive — group = {group}")
    ax.set_xticks([1, 2, 3])
    ax.set_ylim([0.8, 1.0])
    ax.grid(True, alpha=0.3)
    if group == "style":
        ax.set_ylabel("BERTScore F1")

fig.suptitle("BERTScore: Baseline vs Consecutive across rewriting steps — 300q", fontsize=14)
fig.tight_layout()

fig.savefig(OUT_BOTH_PDF)
fig.savefig(OUT_BOTH_PNG, dpi=160)
print(f"Saved: {OUT_BOTH_PDF}")
