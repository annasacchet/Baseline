"""Plot BERTScore Baseline and Consecutive across rewriting steps, by hop count."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "results/300q/rewriting_chains_300q_bertscore.csv"
CHAINS_CSV = REPO / "results/300q/rewriting_chains_300q.csv"

OUT_BASELINE_PDF = REPO / "results/plots/300q/bertscore_baseline_by_hop.pdf"
OUT_BASELINE_PNG = REPO / "results/plots/300q/png/bertscore_baseline_by_hop.png"
OUT_CONSECUTIVE_PDF = REPO / "results/plots/300q/bertscore_consecutive_by_hop.pdf"
OUT_CONSECUTIVE_PNG = REPO / "results/plots/300q/png/bertscore_consecutive_by_hop.png"
OUT_BOTH_PDF = REPO / "results/plots/300q/bertscore_baseline_consecutive_by_hop.pdf"
OUT_BOTH_PNG = REPO / "results/plots/300q/png/bertscore_baseline_consecutive_by_hop.png"

HOP_COLORS = {2: "#1f77b4", 3: "#ff7f0e", 4: "#d62728"}

df = pd.read_csv(CSV)
chains = pd.read_csv(CHAINS_CSV)

# Extract hop count from qid
chains["n_hop"] = chains["qid"].str.extract(r"^(\d+)hop").astype(int)
hop_map = chains[["qid", "n_hop"]].drop_duplicates().set_index("qid")["n_hop"].to_dict()
df["n_hop"] = df["qid"].map(hop_map)

def agg_metric(metric_col):
    return (
        df.groupby(["n_hop", "step"])[metric_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

agg_baseline = agg_metric("bert_f1_baseline")
agg_baseline["sem"] = agg_baseline["std"] / agg_baseline["count"].pow(0.5)

agg_consecutive = agg_metric("bert_f1_consecutive")
agg_consecutive["sem"] = agg_consecutive["std"] / agg_consecutive["count"].pow(0.5)

# ===== Plot 1: Baseline by hop =====
fig, ax = plt.subplots(figsize=(7.5, 5))

for hop in [2, 3, 4]:
    sub = agg_baseline[agg_baseline["n_hop"] == hop].sort_values("step")
    ax.errorbar(
        sub["step"], sub["mean"], yerr=sub["sem"],
        marker="o", linewidth=2, capsize=3,
        label=f"{hop}-hop", color=HOP_COLORS[hop],
    )

ax.set_xlabel("Rewriting step")
ax.set_ylabel("BERTScore F1 Baseline (mean ± SEM)")
ax.set_title("BERTScore Baseline (vs original) by question hop count — 300q")
ax.set_xticks([1, 2, 3])
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()

OUT_BASELINE_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_BASELINE_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_BASELINE_PDF)
fig.savefig(OUT_BASELINE_PNG, dpi=160)
print(f"Saved: {OUT_BASELINE_PDF}")

# ===== Plot 2: Consecutive by hop =====
fig, ax = plt.subplots(figsize=(7.5, 5))

for hop in [2, 3, 4]:
    sub = agg_consecutive[agg_consecutive["n_hop"] == hop].sort_values("step")
    ax.errorbar(
        sub["step"], sub["mean"], yerr=sub["sem"],
        marker="o", linewidth=2, capsize=3,
        label=f"{hop}-hop", color=HOP_COLORS[hop],
    )

ax.set_xlabel("Rewriting step")
ax.set_ylabel("BERTScore F1 Consecutive (mean ± SEM)")
ax.set_title("BERTScore Consecutive (vs previous step) by question hop count — 300q")
ax.set_xticks([1, 2, 3])
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()

fig.savefig(OUT_CONSECUTIVE_PDF)
fig.savefig(OUT_CONSECUTIVE_PNG, dpi=160)
print(f"Saved: {OUT_CONSECUTIVE_PDF}")

# ===== Plot 3: Both side-by-side (top=baseline, bottom=consecutive) =====
fig, axes = plt.subplots(2, 1, figsize=(7.5, 10))

# Top: Baseline
for hop in [2, 3, 4]:
    sub = agg_baseline[agg_baseline["n_hop"] == hop].sort_values("step")
    axes[0].errorbar(
        sub["step"], sub["mean"], yerr=sub["sem"],
        marker="o", linewidth=2, capsize=3,
        label=f"{hop}-hop", color=HOP_COLORS[hop],
    )

axes[0].set_xlabel("Rewriting step")
axes[0].set_ylabel("BERTScore F1 Baseline")
axes[0].set_title("Baseline — similarity to original")
axes[0].set_xticks([1, 2, 3])
axes[0].grid(True, alpha=0.3)
axes[0].legend(frameon=False)

# Bottom: Consecutive
for hop in [2, 3, 4]:
    sub = agg_consecutive[agg_consecutive["n_hop"] == hop].sort_values("step")
    axes[1].errorbar(
        sub["step"], sub["mean"], yerr=sub["sem"],
        marker="o", linewidth=2, capsize=3,
        label=f"{hop}-hop", color=HOP_COLORS[hop],
    )

axes[1].set_xlabel("Rewriting step")
axes[1].set_ylabel("BERTScore F1 Consecutive")
axes[1].set_title("Consecutive — local coherence between steps")
axes[1].set_xticks([1, 2, 3])
axes[1].grid(True, alpha=0.3)
axes[1].legend(frameon=False)

fig.suptitle("BERTScore: Baseline vs Consecutive by question hop count — 300q", fontsize=14)
fig.tight_layout()

fig.savefig(OUT_BOTH_PDF)
fig.savefig(OUT_BOTH_PNG, dpi=160)
print(f"Saved: {OUT_BOTH_PDF}")

# Print tables
print("\n=== BERTScore Baseline by n_hop × step ===")
print(agg_baseline.pivot(index="n_hop", columns="step", values="mean").round(3))
print("\n=== BERTScore Consecutive by n_hop × step ===")
print(agg_consecutive.pivot(index="n_hop", columns="step", values="mean").round(3))
