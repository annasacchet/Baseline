"""Plot mean token length across rewriting steps, one line per instruction_type."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "results/300q/rewriting_chains_300q.csv"
OUT_PDF = REPO / "results/plots/300q/rewriting_tokens_by_step.pdf"
OUT_PNG = REPO / "results/plots/300q/png/rewriting_tokens_by_step.png"

INSTR_ORDER = ["formality", "paraphrase", "shorten", "elaborate"]
COLORS = {
    "formality": "#1f77b4",
    "paraphrase": "#2ca02c",
    "shorten": "#d62728",
    "elaborate": "#9467bd",
}

df = pd.read_csv(CSV)

agg = (
    df.groupby(["instruction_type", "step"])["n_tokens"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
agg["sem"] = agg["std"] / agg["count"].pow(0.5)

fig, ax = plt.subplots(figsize=(7.5, 5))

for instr in INSTR_ORDER:
    sub = agg[agg["instruction_type"] == instr].sort_values("step")
    if sub.empty:
        continue
    ax.errorbar(
        sub["step"],
        sub["mean"],
        yerr=sub["sem"],
        marker="o",
        linewidth=2,
        capsize=3,
        label=instr,
        color=COLORS[instr],
    )

ax.set_xlabel("Rewriting step")
ax.set_ylabel("n_tokens (mean ± SEM)")
ax.set_title("Token length across rewriting steps — 300q")
ax.set_xticks(sorted(df["step"].unique()))
ax.grid(True, alpha=0.3)
ax.legend(title="instruction_type", frameon=False)

fig.tight_layout()
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=160)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
