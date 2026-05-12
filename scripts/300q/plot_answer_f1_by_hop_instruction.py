"""Two plots of Answer F1 across rewriting steps:
  (1) one line per n_hop (2/3/4), averaging over instruction_type
  (2) one line per instruction_type, averaging over groups and hops
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "results/300q/rewriting_chains_300q_answer_f1_olmo31_4bit_SMOKE.csv"

OUT_HOP_PDF = REPO / "results/plots/300q/answer_f1_by_hop_olmo31_4bit.pdf"
OUT_HOP_PNG = REPO / "results/plots/300q/png/answer_f1_by_hop_olmo31_4bit.png"
OUT_INSTR_PDF = REPO / "results/plots/300q/answer_f1_by_instruction_olmo31_4bit.pdf"
OUT_INSTR_PNG = REPO / "results/plots/300q/png/answer_f1_by_instruction_olmo31_4bit.png"

INSTR_ORDER = ["formality", "paraphrase", "shorten", "elaborate"]
INSTR_COLORS = {
    "formality": "#1f77b4",
    "paraphrase": "#2ca02c",
    "shorten": "#d62728",
    "elaborate": "#9467bd",
}
HOP_COLORS = {2: "#1f77b4", 3: "#ff7f0e", 4: "#d62728"}

df = pd.read_csv(CSV)
df["n_hop"] = df["qid"].str.extract(r"^(\d+)hop").astype(int)
steps = sorted(df["step"].unique())

# Step 0 = original question, identical across instruction_type/group/run.
# Keep one row per qid at step 0 so means/SEM aren't inflated by duplicates.
df = pd.concat(
    [df[df["step"] == 0].drop_duplicates(subset="qid"), df[df["step"] != 0]],
    ignore_index=True,
)


# Columns that are intrinsic to the qid (same value for every row of that
# qid) vs. rewriting-axis columns that are meaningless at step 0.
QID_INTRINSIC = {"n_hop"}
REWRITING_AXES = {"instruction_type", "group", "run"}


def agg(g_cols):
    # For rewriting-axis grouping columns, step 0 has a single deduped row per
    # qid and the column value is arbitrary; broadcast it across the labels
    # seen at later steps so step-0 stats appear under each group.
    # For qid-intrinsic columns (e.g. n_hop), keep the original value.
    later = df[df["step"] != 0]
    base0 = df[df["step"] == 0]
    broadcast_cols = [c for c in g_cols if c in REWRITING_AXES]
    if broadcast_cols:
        base0 = base0.drop(columns=broadcast_cols)
        labels = later[broadcast_cols].drop_duplicates()
        base0 = base0.merge(labels, how="cross")
    full = pd.concat([base0, later], ignore_index=True)
    a = full.groupby(g_cols + ["step"])["answer_f1"].agg(["mean", "std", "count"]).reset_index()
    a["sem"] = a["std"] / a["count"].pow(0.5)
    return a


# ---------------- (1) F1 by n_hop ----------------
agg_hop = agg(["n_hop"])

fig, ax = plt.subplots(figsize=(7.5, 5))
for hop in [2, 3, 4]:
    sub = agg_hop[agg_hop["n_hop"] == hop].sort_values("step")
    ax.errorbar(
        sub["step"], sub["mean"], yerr=sub["sem"],
        marker="o", linewidth=2, capsize=3,
        label=f"{hop}-hop", color=HOP_COLORS[hop],
    )
ax.set_xlabel("Rewriting step")
ax.set_ylabel("Answer F1 (mean ± SEM)")
ax.set_title("Answer F1 by question hop count — 300q (OLMo-3.1-32B, 4-bit)")
ax.set_xticks(steps)
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()
OUT_HOP_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_HOP_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_HOP_PDF)
fig.savefig(OUT_HOP_PNG, dpi=160)
print(f"Saved: {OUT_HOP_PDF}")

# ---------------- (2) F1 by instruction_type ----------------
agg_instr = agg(["instruction_type"])

fig, ax = plt.subplots(figsize=(7.5, 5))
for instr in INSTR_ORDER:
    sub = agg_instr[agg_instr["instruction_type"] == instr].sort_values("step")
    ax.errorbar(
        sub["step"], sub["mean"], yerr=sub["sem"],
        marker="o", linewidth=2, capsize=3,
        label=instr, color=INSTR_COLORS[instr],
    )
ax.set_xlabel("Rewriting step")
ax.set_ylabel("Answer F1 (mean ± SEM)")
ax.set_title("Answer F1 by instruction_type — 300q (OLMo-3.1-32B, 4-bit)")
ax.set_xticks(steps)
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, title="instruction_type")
fig.tight_layout()
fig.savefig(OUT_INSTR_PDF)
fig.savefig(OUT_INSTR_PNG, dpi=160)
print(f"Saved: {OUT_INSTR_PDF}")

# ---------------- tables ----------------
print("\n=== Tabella F1 per n_hop × step ===")
print(agg_hop.pivot(index="n_hop", columns="step", values="mean").round(3))
print("\n=== Tabella F1 per instruction_type × step ===")
print(agg_instr.pivot(index="instruction_type", columns="step", values="mean").round(3))
