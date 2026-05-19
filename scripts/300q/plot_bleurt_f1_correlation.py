"""Plot della correlazione tra Answer F1 e bleurt_answer (gold vs predicted)."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
F1_CSV     = REPO / "results/300q/rewriting_chains_300q_answer_f1_olmo31_4bit.csv"
BLEURT_CSV = REPO / "results/300q/rewriting_chains_300q_bleurt.csv"
OUT_DIR    = REPO / "results/plots/300q/png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

key = ["qid", "group", "instruction_type", "run", "step"]

f1 = pd.read_csv(F1_CSV)
bl = pd.read_csv(BLEURT_CSV)
df = f1[key + ["answer_f1", "gold_answer", "predicted_answer"]].merge(
    bl[key + ["bleurt_answer"]], on=key
)
df = df[
    df["predicted_answer"].notna() &
    (df["predicted_answer"].astype(str).str.strip() != "")
].copy()

r, _ = stats.pearsonr(df["answer_f1"], df["bleurt_answer"])

# ── 1. Scatter con densità (hexbin) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
hb = ax.hexbin(df["answer_f1"], df["bleurt_answer"],
               gridsize=50, cmap="YlOrRd", mincnt=1, bins="log")
fig.colorbar(hb, ax=ax, label="log(count)")
ax.set_xlabel("Answer F1")
ax.set_ylabel("BLEURT (gold vs predicted)")
ax.set_title(f"Answer F1 vs BLEURT — 300q\nPearson r = {r:.3f}  (n={len(df):,})")
ax.axvline(0, color="steelblue", lw=0.8, ls="--", alpha=0.6)
ax.axhline(0.3, color="steelblue", lw=0.8, ls="--", alpha=0.6,
           label="BLEURT=0.3 (soglia falsi negativi)")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
out = OUT_DIR / "bleurt_vs_answerf1_scatter.png"
fig.savefig(out, dpi=160)
print(f"Saved: {out}")
plt.close(fig)

# ── 2. Boxplot bleurt per fascia di Answer F1 ────────────────────────────────
bins   = [-0.01, 0.0, 0.25, 0.5, 0.75, 1.01]
labels = ["F1 = 0", "0 < F1 ≤ 0.25", "0.25 < F1 ≤ 0.5",
          "0.5 < F1 ≤ 0.75", "F1 > 0.75"]
df["f1_bin"] = pd.cut(df["answer_f1"], bins=bins, labels=labels)

fig, ax = plt.subplots(figsize=(9, 5))
groups  = [df[df["f1_bin"] == l]["bleurt_answer"].dropna().values for l in labels]
bp = ax.boxplot(groups, patch_artist=True, notch=False,
                medianprops=dict(color="black", lw=2))
palette = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"]
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
counts = [len(g) for g in groups]
ax.set_xticklabels([f"{l}\n(n={c:,})" for l, c in zip(labels, counts)], fontsize=9)
ax.set_ylabel("BLEURT (gold vs predicted)")
ax.set_title("Distribuzione BLEURT per fascia di Answer F1 — 300q")
ax.axhline(0.3, color="grey", lw=0.8, ls="--", alpha=0.7, label="BLEURT=0.3")
ax.axhline(0.1, color="grey", lw=0.8, ls=":",  alpha=0.7, label="BLEURT=0.1 (errore certo)")
ax.legend(fontsize=8, frameon=False)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
out = OUT_DIR / "bleurt_by_f1_bin_boxplot.png"
fig.savefig(out, dpi=160)
print(f"Saved: {out}")
plt.close(fig)

# ── 3. Focus su F1=0: distribuzione bleurt con zone annotate ─────────────────
zeros = df[df["answer_f1"] == 0]["bleurt_answer"].dropna()

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(zeros, bins=60, color="#d62728", alpha=0.75, edgecolor="white", lw=0.3)
ax.axvline(0.1, color="#333", lw=1.5, ls=":", label="0.1 — errore certo (<)")
ax.axvline(0.3, color="#333", lw=1.5, ls="--", label="0.3 — soglia falsi negativi (mid)")
ax.axvline(0.5, color="#333", lw=1.5, ls="-",  label="0.5 — soglia falsi negativi (high)")

# zone colorate
ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] > -0.1 else -0.05,
           0.1,  alpha=0.08, color="black",   label="_nolegend_")
ax.axvspan(0.3, 0.5, alpha=0.08, color="orange", label="_nolegend_")
ax.axvspan(0.5, 1.1, alpha=0.08, color="green",  label="_nolegend_")

n_certain  = (zeros < 0.1).sum()
n_mid      = zeros.between(0.3, 0.5).sum()
n_high     = (zeros >= 0.5).sum()

ax.text(0.05,  ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 100,
        f"errore certo\n{n_certain:,}", ha="center", fontsize=8, color="#333")
ax.text(0.40,  ax.get_ylim()[1]*0.5  if ax.get_ylim()[1] > 0 else 60,
        f"mid\n{n_mid:,}", ha="center", fontsize=8, color="darkorange")
ax.text(0.65,  ax.get_ylim()[1]*0.5  if ax.get_ylim()[1] > 0 else 60,
        f"high\n{n_high:,}", ha="center", fontsize=8, color="darkgreen")

ax.set_xlabel("BLEURT (gold vs predicted)")
ax.set_ylabel("Conteggio")
ax.set_title(f"Distribuzione BLEURT dove Answer F1 = 0 — 300q  (n={len(zeros):,})")
ax.legend(fontsize=8, frameon=False)
ax.set_xlim(-0.05, 1.1)
fig.tight_layout()
out = OUT_DIR / "bleurt_distribution_f1zero.png"
fig.savefig(out, dpi=160)
print(f"Saved: {out}")
plt.close(fig)
