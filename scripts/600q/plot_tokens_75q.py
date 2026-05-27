"""Token length distribution across rewriting steps — 600q run (75 qids so far).

Three figures in results/plots/png/:
  - tokens_by_step_75q.png        : mean tokens per step (line + error bars)
  - tokens_by_step_instr_75q.png  : faceted by instruction_type
  - tokens_boxplot_75q.png        : boxplot per (step, instruction)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
CHAINS = REPO / "rewriting_chains_musique_600q.csv"
OUT_DIR = REPO / "results/plots/png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTR_ORDER = ["paraphrase", "formality", "shorten", "elaborate"]
COLORS = {
    "paraphrase": "#1f77b4",
    "formality":  "#2ca02c",
    "shorten":    "#ff7f0e",
    "elaborate":  "#d62728",
}

df = pd.read_csv(CHAINS)
N_QIDS = df["qid"].nunique()
print(f"Loaded {len(df)} rows, {N_QIDS} qids")

# E0 token length is the baseline (step 0). Compute per-qid baseline.
e0 = df[df["step"] == 0].groupby("qid")["n_tokens"].mean()
print(f"\nE0 (step 0) tokens: mean={e0.mean():.0f}, median={e0.median():.0f}, "
      f"min={e0.min():.0f}, max={e0.max():.0f}")

# Only rewriting steps for trajectories (skip 0 for the per-instruction plots
# because step 0 is the same E0 for every instruction).
df_rw = df[df["step"].isin([1, 2, 3])].copy()

# ────────────────────────────────────────────────────────────────────
# (1) Mean tokens by step — overall, with E0 dashed
# ────────────────────────────────────────────────────────────────────
def step_summary(d):
    g = d.groupby("step")["n_tokens"]
    return pd.DataFrame({
        "mean": g.mean(),
        "se":   g.std(ddof=1) / np.sqrt(g.count()),
        "n":    g.count(),
    })

s_all = step_summary(df)
print("\nTokens by step (all instructions):")
print(s_all.round(1))

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar([1, 2, 3], s_all.loc[[1,2,3], "mean"], yerr=s_all.loc[[1,2,3], "se"],
            fmt="o-", color="#1f77b4", lw=2.5, capsize=5, ms=9,
            label=f"mean rewritten text  (n={N_QIDS} qids × 12 chains)")
ax.axhline(s_all.loc[0, "mean"], color="black", ls="--", lw=1.2,
           label=f"E0 baseline = {s_all.loc[0,'mean']:.0f} tokens")
for st in [1, 2, 3]:
    ax.text(st, s_all.loc[st, "mean"] + 25,
            f"{s_all.loc[st, 'mean']:.0f}", ha="center", fontsize=10, color="#1f77b4")
ax.set_xticks([1, 2, 3])
ax.set_xlabel("Rewriting step", fontsize=11)
ax.set_ylabel("Mean n_tokens (OLMo tokenizer)", fontsize=11)
ax.set_title(f"Rewritten text length across steps — 600q (75 qids done)\n"
             "All instructions pooled", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, frameon=False, loc="best")
fig.tight_layout()
out = OUT_DIR / "tokens_by_step_75q.png"
fig.savefig(out, dpi=160)
print(f"Saved: {out}")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────
# (2) Mean tokens by step × instruction — all four lines in one plot
# ────────────────────────────────────────────────────────────────────
def step_instr_summary(d):
    g = d.groupby(["instruction_type", "step"])["n_tokens"]
    return pd.DataFrame({"mean": g.mean(),
                         "se":   g.std(ddof=1) / np.sqrt(g.count()),
                         "n":    g.count()})

s_inst = step_instr_summary(df_rw)
print("\nTokens by step × instruction:")
print(s_inst.round(1))

fig, ax = plt.subplots(figsize=(9, 5.5))
for instr in INSTR_ORDER:
    if instr not in s_inst.index.get_level_values(0):
        continue
    sub = s_inst.loc[instr].reindex([1, 2, 3])
    ax.errorbar([1, 2, 3], sub["mean"], yerr=sub["se"], fmt="o-",
                color=COLORS[instr], lw=2.5, capsize=4, ms=8, label=instr)
ax.axhline(s_all.loc[0, "mean"], color="black", ls="--", lw=1.2,
           label=f"E0 baseline = {s_all.loc[0,'mean']:.0f} tok")
ax.set_xticks([1, 2, 3])
ax.set_xlabel("Rewriting step", fontsize=11)
ax.set_ylabel("Mean n_tokens", fontsize=11)
ax.set_title(f"Rewritten text length by instruction — 600q (75 qids)\n"
             "elaborate expands above E0; shorten compresses; others stay close",
             fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, frameon=False, loc="best")
fig.tight_layout()
out = OUT_DIR / "tokens_by_step_instr_75q.png"
fig.savefig(out, dpi=160)
print(f"Saved: {out}")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────
# (3) Boxplot per (step, instruction) — distribution, not just mean
# ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
positions = []
data = []
colors = []
labels = []
gap = 0.6
for i, instr in enumerate(INSTR_ORDER):
    for s in [1, 2, 3]:
        sub = df_rw[(df_rw["instruction_type"] == instr) & (df_rw["step"] == s)]["n_tokens"].values
        positions.append(i * (3 + gap) + s)
        data.append(sub)
        colors.append(COLORS[instr])
        labels.append(f"{instr}\nstep {s}" if s == 2 else f"step {s}")

bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                medianprops=dict(color="black", lw=1.6),
                flierprops=dict(marker="o", markersize=3, alpha=0.3))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.65)

ax.axhline(s_all.loc[0, "mean"], color="black", ls="--", lw=1.2,
           label=f"E0 baseline = {s_all.loc[0,'mean']:.0f} tok")
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("n_tokens (rewritten text)", fontsize=11)
ax.set_title(f"Distribution of rewritten-text length — 600q (75 qids)\n"
             "Each box: 75 qids × 3 wordings = 225 chains",
             fontsize=12)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=10, frameon=False, loc="upper right")
fig.tight_layout()
out = OUT_DIR / "tokens_boxplot_75q.png"
fig.savefig(out, dpi=160)
print(f"Saved: {out}")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────
# Compression / expansion ratio summary
# ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPRESSION / EXPANSION ratio (step / E0)")
print("=" * 70)
merged = df_rw.merge(e0.rename("e0_tokens"), on="qid")
merged["ratio"] = merged["n_tokens"] / merged["e0_tokens"]
piv = merged.groupby(["instruction_type", "step"])["ratio"].mean().unstack()
piv = piv.reindex(INSTR_ORDER)
print((piv * 100).round(1).to_string())
print("\nValues are % of E0 length (100 = same length as source).")
