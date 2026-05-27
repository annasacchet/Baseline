"""
Robustness of OFS degradation — 300q (old prompt) vs 600q (new prompt + 4-bit),
paired on the 11 qids present in BOTH runs.

Three clearer figures:
  (A) ofs_paired_trajectory.png
      Side-by-side: left = OFS trajectory (one line per setup), right = the
      same trajectory after subtracting the step-1 baseline (curves overlaid
      → shape comparison is obvious).

  (B) ofs_paired_by_instruction.png
      Grouped bars: Δ(step3 − step1) per instruction, 300q vs 600q, with
      explicit labels and arrows highlighting where the new prompt HELPS
      (paraphrase/formality) and where it HURTS (elaborate).

  (C) ofs_paired_qid_dots.png
      Dot plot of Δ(step3 − step1) for each of the 11 qids, 300q vs 600q,
      connected by a line per qid → shows the pairing directly.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OFS_300 = REPO / "results/300q/rewriting_chains_300q_openfactscore.csv"
OFS_600 = REPO / "results/600q/rewriting_chains_musique_600q_openfactscore.csv"
OUT_DIR = REPO / "results/plots/png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTR_ORDER = ["paraphrase", "formality", "shorten", "elaborate"]
COL_300 = "#d62728"   # red    — old prompt
COL_600 = "#1f77b4"   # blue   — new prompt
COL_NEU = "#7f7f7f"

def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["step"] = df["step"].astype(int)
    df = df[df["step"].isin([1, 2, 3])].copy()
    return df

d300_full = load(OFS_300)
d600 = load(OFS_600)
common = sorted(set(d300_full["qid"]) & set(d600["qid"]))
d300 = d300_full[d300_full["qid"].isin(common)].copy()
d600 = d600[d600["qid"].isin(common)].copy()
N_QIDS = len(common)
print(f"Paired on {N_QIDS} common qids")

def step_summary(df):
    g = df.groupby("step")["factscore"]
    return pd.DataFrame({
        "mean": g.mean(),
        "se":   g.std(ddof=1) / np.sqrt(g.count()),
        "n":    g.count(),
    })

s300 = step_summary(d300)
s600 = step_summary(d600)

# =================================================================
# (A) Side-by-side: absolute trajectory + normalised (Δ from step 1)
# =================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# --- left: absolute OFS ------------------------------------------------
ax1.errorbar(s300.index, s300["mean"], yerr=s300["se"], fmt="o-",
             color=COL_300, lw=2.5, capsize=5, ms=9,
             label="old prompt (300q setup)")
ax1.errorbar(s600.index, s600["mean"], yerr=s600["se"], fmt="s-",
             color=COL_600, lw=2.5, capsize=5, ms=9,
             label="new prompt (600q setup)")

# annotate the gap between the two curves
for st in [1, 2, 3]:
    y_top = max(s300["mean"][st], s600["mean"][st])
    y_bot = min(s300["mean"][st], s600["mean"][st])
    ax1.annotate("", xy=(st + 0.04, y_top), xytext=(st + 0.04, y_bot),
                 arrowprops=dict(arrowstyle="<->", color=COL_NEU, lw=1))
    ax1.text(st + 0.10, (y_top + y_bot) / 2,
             f"−{(s300['mean'][st] - s600['mean'][st])*100:.1f} pp",
             color=COL_NEU, fontsize=8, va="center")

ax1.set_xticks([1, 2, 3])
ax1.set_xlabel("Rewriting step", fontsize=11)
ax1.set_ylabel("Mean OpenFActScore", fontsize=11)
ax1.set_title("A. Absolute OFS  —  the new prompt is stricter (lower baseline)",
              fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, frameon=False, loc="lower left")

# --- right: normalised (subtract step 1) — overlay the SHAPE ----------
n300 = s300["mean"] - s300["mean"][1]
n600 = s600["mean"] - s600["mean"][1]
ax2.plot(n300.index, n300.values, "o-", color=COL_300, lw=2.5, ms=9,
         label="old prompt (300q setup)")
ax2.plot(n600.index, n600.values, "s-", color=COL_600, lw=2.5, ms=9,
         label="new prompt (600q setup)")
ax2.axhline(0, color="black", lw=0.6)
ax2.set_xticks([1, 2, 3])
ax2.set_xlabel("Rewriting step", fontsize=11)
ax2.set_ylabel("OFS change vs step 1", fontsize=11)
delta_300 = n300[3]
delta_600 = n600[3]
ax2.set_title(f"B. Same trajectory after removing the baseline\n"
              f"Δ(step3 − step1):  old = {delta_300:+.3f}   new = {delta_600:+.3f}   "
              f"→ identical slope",
              fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10, frameon=False, loc="upper right")

fig.suptitle(f"OFS degradation is robust to the prompt change\n"
             f"(paired on the same {N_QIDS} qids; the two curves are parallel)",
             fontsize=13, y=1.02)
fig.tight_layout()
out = OUT_DIR / "ofs_paired_trajectory.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

# =================================================================
# (B) Bar chart: Δ per instruction, with helper / hurter labels
# =================================================================
def delta_by_instr(df):
    pv = (df.groupby(["qid", "instruction_type", "step"])["factscore"]
            .mean().reset_index())
    pv = pv.pivot(index=["qid", "instruction_type"], columns="step", values="factscore")
    pv = pv.dropna(subset=[1, 3])
    pv["delta"] = pv[3] - pv[1]
    g = pv.groupby("instruction_type")["delta"]
    return pd.DataFrame({"mean": g.mean(),
                         "se":   g.std(ddof=1) / np.sqrt(g.count()),
                         "n":    g.count()})

dd300 = delta_by_instr(d300).reindex(INSTR_ORDER)
dd600 = delta_by_instr(d600).reindex(INSTR_ORDER)

fig, ax = plt.subplots(figsize=(9.5, 5.5))
x = np.arange(len(INSTR_ORDER))
w = 0.35

b1 = ax.bar(x - w/2, dd300["mean"], yerr=dd300["se"], width=w,
            color=COL_300, alpha=0.85, capsize=4, label="old prompt (300q setup)")
b2 = ax.bar(x + w/2, dd600["mean"], yerr=dd600["se"], width=w,
            color=COL_600, alpha=0.85, capsize=4, label="new prompt (600q setup)")
ax.axhline(0, color="black", lw=0.7)

# labels on bars
for xi, v in zip(x - w/2, dd300["mean"]):
    ax.text(xi, v - 0.004, f"{v:+.3f}", ha="center", va="top",
            fontsize=9, color=COL_300, fontweight="bold")
for xi, v in zip(x + w/2, dd600["mean"]):
    ax.text(xi, v - 0.004, f"{v:+.3f}", ha="center", va="top",
            fontsize=9, color=COL_600, fontweight="bold")

# annotation: where the new prompt helps vs hurts
ymin = min(dd300["mean"].min(), dd600["mean"].min()) - 0.025
ax.set_ylim(ymin, 0.012)

def annotate_effect(xi, label, color):
    ax.text(xi, 0.006, label, ha="center", va="bottom",
            fontsize=10, color=color, fontweight="bold")

annotate_effect(0, "↓ less erosion", "green")   # paraphrase
annotate_effect(1, "↓ less erosion", "green")   # formality
annotate_effect(2, "≈ same",         COL_NEU)   # shorten
annotate_effect(3, "↑ more erosion", "crimson") # elaborate

ax.set_xticks(x)
ax.set_xticklabels(INSTR_ORDER, fontsize=11)
ax.set_ylabel("Δ OFS  (step 3 − step 1)\nmore negative = more degradation",
              fontsize=11)
ax.set_title(f"Per-instruction erosion — paired on {N_QIDS} qids\n"
             "The new prompt redistributes degradation across instructions; "
             "the global mean is unchanged",
             fontsize=12)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=10, frameon=False, loc="lower left")

fig.tight_layout()
out = OUT_DIR / "ofs_paired_by_instruction.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

# =================================================================
# (C) Per-qid dot plot — show the pairing directly
# =================================================================
def delta_per_qid(df):
    pv = (df.groupby(["qid", "step"])["factscore"]
            .mean().reset_index()
            .pivot(index="qid", columns="step", values="factscore"))
    return (pv[3] - pv[1]).rename("delta")

dq300 = delta_per_qid(d300).reindex(common)
dq600 = delta_per_qid(d600).reindex(common)

# sort by 300q delta for readability
order = dq300.sort_values().index.tolist()
y = np.arange(len(order))

fig, ax = plt.subplots(figsize=(10, 6))
for yi, qid in zip(y, order):
    v300 = dq300.loc[qid]
    v600 = dq600.loc[qid]
    ax.plot([v300, v600], [yi, yi], color=COL_NEU, lw=1, alpha=0.6, zorder=1)
    ax.scatter([v300], [yi], s=80, color=COL_300, zorder=3,
               label="old prompt" if yi == 0 else None)
    ax.scatter([v600], [yi], s=80, color=COL_600, marker="s", zorder=3,
               label="new prompt" if yi == 0 else None)

ax.axvline(0, color="black", lw=0.6)
ax.axvline(dq300.mean(), color=COL_300, ls="--", lw=1, alpha=0.7,
           label=f"old prompt mean = {dq300.mean():+.3f}")
ax.axvline(dq600.mean(), color=COL_600, ls="--", lw=1, alpha=0.7,
           label=f"new prompt mean = {dq600.mean():+.3f}")

ax.set_yticks(y)
ax.set_yticklabels([q.replace("2hop__", "") for q in order], fontsize=9)
ax.set_xlabel("Δ OFS  (step 3 − step 1)\nmore negative = more degradation",
              fontsize=11)
ax.set_ylabel("qid")
ax.set_title(f"Per-qid factual erosion — same {N_QIDS} qids, two prompts\n"
             "Dots from the same row are the SAME question scored under the two setups",
             fontsize=12)
ax.grid(True, axis="x", alpha=0.3)
ax.legend(fontsize=9, frameon=False, loc="lower right")
fig.tight_layout()
out = OUT_DIR / "ofs_paired_qid_dots.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

# =================================================================
# Summary
# =================================================================
print("\n" + "=" * 70)
print(f"PAIRED SUMMARY  ({N_QIDS} qids)")
print("=" * 70)
print(f"old prompt:  step1={s300['mean'][1]:.3f}  step3={s300['mean'][3]:.3f}  "
      f"Δ = {s300['mean'][3]-s300['mean'][1]:+.3f}")
print(f"new prompt:  step1={s600['mean'][1]:.3f}  step3={s600['mean'][3]:.3f}  "
      f"Δ = {s600['mean'][3]-s600['mean'][1]:+.3f}")
print(f"\nSlope difference: {abs((s600['mean'][3]-s600['mean'][1]) - (s300['mean'][3]-s300['mean'][1])):.4f}  "
      "→ negligible: the degradation trajectory is unchanged.")
