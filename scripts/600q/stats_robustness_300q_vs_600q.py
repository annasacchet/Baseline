"""Statistical tests for trajectory robustness — 300q vs 600q (paired on common qids).

Three tests answer three different questions:

(1) Is OFS degradation step-by-step statistically significant in BOTH setups?
    → paired t-test on Δ(step3 − step1) per qid×instruction, against 0.

(2) Is the *slope* of degradation different between setups?
    → paired t-test on (Δ_300q − Δ_600q) per qid×instruction, against 0.
    H0: same slope. Reject only if the two trajectories actually differ.

(3) Is the *ranking* across instructions preserved?
    → Spearman rank correlation between the per-instruction Δ vectors.
    → also report Kendall's tau on instruction-level means.

We also report the same battery for token compression ratio.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
OFS_300 = REPO / "results/300q/rewriting_chains_300q_openfactscore.csv"
OFS_600 = REPO / "results/600q/rewriting_chains_musique_600q_openfactscore.csv"
CH_300  = REPO / "results/300q/rewriting_chains_300q.csv"
CH_600  = REPO / "rewriting_chains_musique_600q.csv"

INSTR_ORDER = ["paraphrase", "formality", "shorten", "elaborate"]

def load_ofs(p):
    d = pd.read_csv(p)
    d["step"] = d["step"].astype(int)
    return d[d["step"].isin([1, 2, 3])].copy()

o300 = load_ofs(OFS_300)
o600 = load_ofs(OFS_600)
common_ofs = sorted(set(o300["qid"]) & set(o600["qid"]))
o300 = o300[o300["qid"].isin(common_ofs)]
o600 = o600[o600["qid"].isin(common_ofs)]
print(f"OFS paired on {len(common_ofs)} qids")

def delta_per_qid_instr(df, metric="factscore"):
    pv = (df.groupby(["qid", "instruction_type", "step"])[metric]
            .mean().reset_index())
    pv = pv.pivot(index=["qid", "instruction_type"], columns="step", values=metric)
    pv = pv.dropna(subset=[1, 3])
    pv["delta"] = pv[3] - pv[1]
    return pv["delta"]

d300 = delta_per_qid_instr(o300)
d600 = delta_per_qid_instr(o600)

print(f"\nPaired observations (qid × instruction): "
      f"300q n={len(d300)}, 600q n={len(d600)}")

# ────────────────────────────────────────────────────────────────────
# (1) Is degradation significant in each setup?
# ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("(1) Is OFS degradation step1→step3 significantly negative?")
print("=" * 70)
for name, d in [("300q (old prompt)", d300), ("600q (new prompt)", d600)]:
    t, p = stats.ttest_1samp(d.values, 0.0)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    cohen_d = m / d.std(ddof=1)
    print(f"{name}:  n={n}  Δ={m:+.4f} (SE={se:.4f})  "
          f"t({n-1})={t:.2f}  p={p:.3g}  Cohen d={cohen_d:.3f}")

# ────────────────────────────────────────────────────────────────────
# (2) Is the slope DIFFERENT between setups? (paired test)
# ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("(2) Paired test: is the slope Δ_300q − Δ_600q different from 0?")
print("=" * 70)
joined = pd.concat([d300.rename("delta_300"), d600.rename("delta_600")], axis=1).dropna()
diff = joined["delta_300"] - joined["delta_600"]
t, p = stats.ttest_rel(joined["delta_300"], joined["delta_600"])
w_stat, w_p = stats.wilcoxon(joined["delta_300"], joined["delta_600"])
print(f"n paired={len(joined)}  "
      f"mean(Δ_300q−Δ_600q)={diff.mean():+.4f}  SD={diff.std(ddof=1):.4f}")
print(f"  paired t-test:  t={t:.2f}  p={p:.3g}")
print(f"  Wilcoxon signed-rank:  W={w_stat:.1f}  p={w_p:.3g}")
print("  → if p > 0.05, the slope is statistically INDISTINGUISHABLE between setups.")

# TOST equivalence test (margin = 0.02 OFS, one third of the typical Δ)
print("\n   TOST equivalence test (margin ±0.02 OFS units):")
margin = 0.02
diff_mean = diff.mean()
diff_se = diff.std(ddof=1) / np.sqrt(len(diff))
t_low  = (diff_mean - (-margin)) / diff_se
t_high = ((+margin) - diff_mean) / diff_se
p_low  = 1 - stats.t.cdf(t_low,  len(diff)-1)
p_high = 1 - stats.t.cdf(t_high, len(diff)-1)
p_tost = max(p_low, p_high)
print(f"   p(equivalence) = {p_tost:.3g}   "
      f"→ if p < 0.05, slopes are statistically EQUIVALENT within ±{margin}.")

# ────────────────────────────────────────────────────────────────────
# (3) Is the RANKING across instructions preserved?
# ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("(3) Is the ranking across instructions preserved?")
print("=" * 70)
mean_300 = d300.groupby("instruction_type").mean().reindex(INSTR_ORDER)
mean_600 = d600.groupby("instruction_type").mean().reindex(INSTR_ORDER)
print("Per-instruction Δ(step3−step1):")
print(pd.DataFrame({"300q": mean_300, "600q": mean_600}).round(4))
rho, p_rho = stats.spearmanr(mean_300, mean_600)
tau, p_tau = stats.kendalltau(mean_300, mean_600)
print(f"\nSpearman ρ = {rho:+.3f}  p={p_rho:.3g}")
print(f"Kendall τ  = {tau:+.3f}  p={p_tau:.3g}")
print("→ ρ ≈ 1 means: same ordering of instructions by degradation strength.")

# Same battery on token compression --------------------------------
print("\n\n" + "=" * 70)
print("SAME TESTS ON TOKEN COMPRESSION RATIO")
print("=" * 70)

def load_chains_with_ratio(p):
    d = pd.read_csv(p)
    e0 = d[d["step"]==0].groupby("qid")["n_tokens"].mean().rename("e0")
    d = d.merge(e0, on="qid")
    d["ratio"] = d["n_tokens"] / d["e0"]
    return d[d["step"].isin([1,2,3])].copy()

c300 = load_chains_with_ratio(CH_300)
c600 = load_chains_with_ratio(CH_600)
common_ch = sorted(set(c300["qid"]) & set(c600["qid"]))
c300 = c300[c300["qid"].isin(common_ch)]
c600 = c600[c600["qid"].isin(common_ch)]
print(f"Chains paired on {len(common_ch)} qids")

dt300 = delta_per_qid_instr(c300, "ratio")
dt600 = delta_per_qid_instr(c600, "ratio")

print("\n(1) Compression step1→step3 < 0 in each setup?")
for name, d in [("300q", dt300), ("600q", dt600)]:
    t, p = stats.ttest_1samp(d.values, 0.0)
    print(f"  {name}:  n={len(d)}  Δratio={d.mean():+.4f}  t={t:.2f}  p={p:.3g}")

print("\n(2) Paired Δ_300q − Δ_600q (token ratio):")
joined_t = pd.concat([dt300.rename("d300"), dt600.rename("d600")], axis=1).dropna()
t, p = stats.ttest_rel(joined_t["d300"], joined_t["d600"])
print(f"  n={len(joined_t)}  mean diff={ (joined_t['d300']-joined_t['d600']).mean():+.4f}  "
      f"t={t:.2f}  p={p:.3g}")

print("\n(3) Ranking preservation across instructions (token ratio Δ):")
m300 = dt300.groupby("instruction_type").mean().reindex(INSTR_ORDER)
m600 = dt600.groupby("instruction_type").mean().reindex(INSTR_ORDER)
print(pd.DataFrame({"300q":m300, "600q":m600}).round(4))
rho, p_rho = stats.spearmanr(m300, m600)
tau, p_tau = stats.kendalltau(m300, m600)
print(f"  Spearman ρ = {rho:+.3f}  p={p_rho:.3g}")
print(f"  Kendall τ  = {tau:+.3f}  p={p_tau:.3g}")
