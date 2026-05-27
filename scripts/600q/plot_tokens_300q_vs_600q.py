"""Token compression: 300q (old prompt) vs 600q (new prompt) by instruction.

Two figures:
  - tokens_compression_compare.png   : compression % of E0 per (step, instr)
                                       full 300q (297 qids) vs full 600q (75 qids)
  - tokens_compression_matched.png   : paired on the 11 qids that exist in both
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
CH_300 = REPO / "results/300q/rewriting_chains_300q.csv"
CH_600 = REPO / "rewriting_chains_musique_600q.csv"
OUT_DIR = REPO / "results/plots/png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTR_ORDER = ["paraphrase", "formality", "shorten", "elaborate"]
COLORS = {
    "paraphrase": "#1f77b4",
    "formality":  "#2ca02c",
    "shorten":    "#ff7f0e",
    "elaborate":  "#d62728",
}

def add_ratio(df):
    e0 = df[df["step"] == 0].groupby("qid")["n_tokens"].mean().rename("e0_tokens")
    df = df.merge(e0, on="qid")
    df["ratio"] = df["n_tokens"] / df["e0_tokens"]
    return df[df["step"].isin([1, 2, 3])].copy()

d300 = add_ratio(pd.read_csv(CH_300))
d600 = add_ratio(pd.read_csv(CH_600))
print(f"300q: {d300['qid'].nunique()} qids | 600q: {d600['qid'].nunique()} qids")

def compression_table(df):
    g = df.groupby(["instruction_type", "step"])["ratio"].mean().unstack().reindex(INSTR_ORDER)
    return (g * 100).round(1)

c300 = compression_table(d300)
c600 = compression_table(d600)
print("\nCompression % of E0  — 300q (old prompt):\n", c300)
print("\nCompression % of E0  — 600q (new prompt, 75 qids):\n", c600)

# ────────────────────────────────────────────────────────────────────
# (1) Lines: 4 instructions × 2 setups, % of E0 across steps
# ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, df, title in [
    (ax1, d300, f"300q (old prompt) — {d300['qid'].nunique()} qids"),
    (ax2, d600, f"600q (new prompt) — {d600['qid'].nunique()} qids"),
]:
    for instr in INSTR_ORDER:
        sub = df[df["instruction_type"] == instr].groupby("step")["ratio"].mean() * 100
        ax.plot(sub.index, sub.values, "o-", color=COLORS[instr], lw=2.5, ms=8, label=instr)
    ax.axhline(100, color="black", ls="--", lw=1, alpha=0.5, label="E0 baseline (100%)")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Rewriting step", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
ax1.set_ylabel("Rewritten length as % of E0", fontsize=11)
ax2.legend(fontsize=10, frameon=False, loc="upper right")

fig.suptitle("Compression by instruction — 300q vs 600q\n"
             "Same ranking (shorten ≪ paraphrase < formality < elaborate); "
             "new prompt compresses LESS at every step",
             fontsize=12, y=1.02)
fig.tight_layout()
out = OUT_DIR / "tokens_compression_compare.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────
# (2) Paired on the 11 common qids
# ────────────────────────────────────────────────────────────────────
common = sorted(set(d300["qid"]) & set(d600["qid"]))
print(f"\nCommon qids: {len(common)}")
m300 = d300[d300["qid"].isin(common)]
m600 = d600[d600["qid"].isin(common)]

cm300 = compression_table(m300)
cm600 = compression_table(m600)
print("\nCompression % of E0  — 300q (matched 11 qids):\n", cm300)
print("\nCompression % of E0  — 600q (matched 11 qids):\n", cm600)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(INSTR_ORDER))
w = 0.11
for j, st in enumerate([1, 2, 3]):
    off_old = -1.5 * w + j * w
    off_new = -1.5 * w + (j + 3) * w + 0.05
    vals_old = [cm300.loc[i, st] for i in INSTR_ORDER]
    vals_new = [cm600.loc[i, st] for i in INSTR_ORDER]
    ax.bar(x + off_old, vals_old, width=w, color="#d62728",
           alpha=0.45 + 0.20 * j, edgecolor="white", lw=0.5,
           label=f"300q step {st}")
    ax.bar(x + off_new, vals_new, width=w, color="#1f77b4",
           alpha=0.45 + 0.20 * j, edgecolor="white", lw=0.5,
           label=f"600q step {st}")

ax.axhline(100, color="black", ls="--", lw=1, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(INSTR_ORDER, fontsize=11)
ax.set_ylabel("Rewritten length as % of E0", fontsize=11)
ax.set_title(f"Compression paired on {len(common)} qids  —  300q (red) vs 600q (blue)\n"
             "Same ordering across instructions; new prompt is consistently less compressive",
             fontsize=12)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
fig.tight_layout()
out = OUT_DIR / "tokens_compression_matched.png"
fig.savefig(out, dpi=160)
print(f"Saved: {out}")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────
# Ranking summary
# ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RANKING by compression (step 3, most → least compressed)")
print("=" * 70)
print("300q (full):", c300[3].sort_values().to_dict())
print("600q (75q): ", c600[3].sort_values().to_dict())
print("\n300q (matched 11):", cm300[3].sort_values().to_dict())
print("600q (matched 11):", cm600[3].sort_values().to_dict())
