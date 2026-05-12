"""Plot Answer F1 across rewriting steps, one line per instruction_type, split by group."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "results/300q/rewriting_chains_300q_answer_f1_olmo31_4bit_SMOKE.csv"
OUT_PDF = REPO / "results/plots/300q/answer_f1_by_step_olmo31_4bit.pdf"
OUT_PNG = REPO / "results/plots/300q/png/answer_f1_by_step_olmo31_4bit.png"

INSTR_ORDER = ["formality", "paraphrase", "shorten", "elaborate"]
COLORS = {
    "formality": "#1f77b4",
    "paraphrase": "#2ca02c",
    "shorten": "#d62728",
    "elaborate": "#9467bd",
}

df = pd.read_csv(CSV)

agg = (
    df.groupby(["group", "instruction_type", "step"])["answer_f1"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
agg["sem"] = agg["std"] / agg["count"].pow(0.5)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, group in zip(axes, ["style", "content"]):
    for instr in INSTR_ORDER:
        sub = agg[(agg["group"] == group) & (agg["instruction_type"] == instr)].sort_values("step")
        if sub.empty:
            continue
        ax.errorbar(
            sub["step"], sub["mean"], yerr=sub["sem"],
            marker="o", linewidth=2, capsize=3,
            label=instr, color=COLORS[instr],
        )
    ax.set_xlabel("Rewriting step")
    ax.set_title(f"group = {group}")
    ax.set_xticks(sorted(df["step"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

axes[0].set_ylabel("Answer F1 (mean ± SEM)")
fig.suptitle("Answer F1 across rewriting steps — 300q (OLMo-3.1-32B-Instruct, 4-bit)")
fig.tight_layout()

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=160)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
