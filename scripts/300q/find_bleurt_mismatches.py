"""
Analisi della correlazione tra Answer F1 e bleurt_answer (gold vs predicted)
per identificare falsi negativi di Answer F1 — risposte semanticamente corrette
ma non matchate lessicalmente.

Sezioni:
  1. Copertura del dataset (step 0 vs step 1-3)
  2. Correlazione globale Answer F1 ~ bleurt_answer (step 1-3)
  3. Distribuzione bleurt_answer per fascia di Answer F1
  4. Stima dell'impatto dei falsi negativi sulle conclusioni
  5. Candidati falsi negativi → bleurt_mismatch_candidates.csv

Criteri per i candidati:
  exact_norm    — gold == predicted dopo normalizzazione punteggiatura/spazi
  gold_in_pred  — gold è sottostringa del predicted (risposta verbosa)
  bleurt_high   — bleurt_answer >= 0.5
  bleurt_mid    — bleurt_answer in [0.3, 0.5) con gold corta (<=25 char)
                  cattura: "10-year"/"10 years", "GDR"/"DDR", "Polish state"/"Poland"

I casi con predicted NaN/vuoto sono esclusi (non sono falsi negativi).
Output: results/300q/bleurt_mismatch_candidates.csv  (colonna `correct` per revisione)
"""

from pathlib import Path
import re
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
F1_CSV     = REPO / "results/300q/rewriting_chains_300q_answer_f1_olmo31_4bit.csv"
BLEURT_CSV = REPO / "results/300q/rewriting_chains_300q_bleurt.csv"
OUT_CSV    = REPO / "results/300q/bleurt_mismatch_candidates.csv"

BLEURT_HIGH      = 0.5
BLEURT_MID       = 0.3
SHORT_GOLD       = 25   # char — soglia gold corta per criterio mid
TRUE_ZERO_BLEURT = 0.1  # sotto questa soglia: errore reale certo

key = ["qid", "group", "instruction_type", "run", "step"]

f1 = pd.read_csv(F1_CSV)
bl = pd.read_csv(BLEURT_CSV)

# ── 1. Copertura ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. COPERTURA DEL DATASET")
print("=" * 60)

# BLEURT è calcolato solo su step 1-3 (le riscritture).
# Step 0 = baseline prima della riscrittura, predicted unico per qid, nessun BLEURT.
step0 = f1[f1["step"] == 0].drop_duplicates(subset="qid")
step0_valid = step0[
    step0["predicted_answer"].notna() &
    (step0["predicted_answer"].astype(str).str.strip() != "")
]

df = f1[f1["step"] > 0].merge(bl[key + ["bleurt_answer"]], on=key, how="inner")
has_pred = df["predicted_answer"].notna() & (df["predicted_answer"].astype(str).str.strip() != "")
full = df[has_pred].copy()

print(f"qid distinte nel dataset:           {f1['qid'].nunique()}")
print(f"Righe totali F1 CSV:                {len(f1):,}")
print(f"Step 0 (baseline, unici per qid):   {len(step0_valid)}  — senza BLEURT")
print(f"Step 1-3 con BLEURT e predicted:    {len(full):,}  — base dell'analisi")
print(f"\nAnswer F1 medio step 0 (baseline):  {step0_valid['answer_f1'].mean():.4f}")
print(f"Answer F1 medio step 1-3:           {full['answer_f1'].mean():.4f}  "
      f"(Δ={full['answer_f1'].mean()-step0_valid['answer_f1'].mean():+.4f})")

# ── 2. Correlazione globale ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. CORRELAZIONE Answer F1 ~ bleurt_answer (step 1-3)")
print("=" * 60)

r, p    = stats.pearsonr(full["answer_f1"], full["bleurt_answer"])
rho, p2 = stats.spearmanr(full["answer_f1"], full["bleurt_answer"])
print(f"Pearson  r={r:.3f}  p={p:.2e}  (n={len(full):,})")
print(f"Spearman ρ={rho:.3f}  p={p2:.2e}")

# ── 3. Distribuzione per fascia di Answer F1 ─────────────────────────────────
print("\n" + "=" * 60)
print("3. BLEURT PER FASCIA DI Answer F1")
print("=" * 60)

bins   = [-0.01, 0.0, 0.25, 0.5, 0.75, 1.01]
labels = ["=0", "0–0.25", "0.25–0.5", "0.5–0.75", "0.75–1"]
full["f1_bin"] = pd.cut(full["answer_f1"], bins=bins, labels=labels)
print(full.groupby("f1_bin", observed=True)["bleurt_answer"]
      .agg(["mean", "median", "count"])
      .round(3).to_string())

zeros_all = full[full["answer_f1"] == 0]
print(f"\nDistribuzione bleurt_answer dove F1=0 (n={len(zeros_all):,}):")
for p_val in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{p_val:2d}: {zeros_all['bleurt_answer'].quantile(p_val/100):.3f}")
print(f"  Sotto {TRUE_ZERO_BLEURT} (errore certo): "
      f"{(zeros_all['bleurt_answer'] < TRUE_ZERO_BLEURT).sum():,} "
      f"({(zeros_all['bleurt_answer'] < TRUE_ZERO_BLEURT).mean()*100:.1f}%)")

# ── 4. Impatto falsi negativi sulle conclusioni ───────────────────────────────
print("\n" + "=" * 60)
print("4. IMPATTO DEI FALSI NEGATIVI")
print("=" * 60)

n_total = len(full)
zeros_f = full[full["answer_f1"] == 0].copy()

def normalize(s: str) -> str:
    s = re.sub(r"[–—]", "-", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

zeros_f["gn"] = zeros_f["gold_answer"].astype(str).apply(normalize)
zeros_f["pn"] = zeros_f["predicted_answer"].astype(str).apply(normalize)
zeros_f["gold_len"] = zeros_f["gold_answer"].astype(str).str.len()

n_exact  = (zeros_f["gn"] == zeros_f["pn"]).sum()
n_cert   = (zeros_f["bleurt_answer"] >= 0.7).sum()
n_high   = (zeros_f["bleurt_answer"] >= 0.5).sum()
n_mid    = (zeros_f["bleurt_answer"].between(0.3, 0.5) & (zeros_f["gold_len"] <= SHORT_GOLD)).sum()
est_mid  = round(n_high * 0.27)  # precision stimata ~27% per fascia 0.5

mean_now = full["answer_f1"].mean()
full_corr = full.copy()
full_corr.loc[
    (full_corr["answer_f1"] == 0) & (full_corr["bleurt_answer"] >= 0.7), "answer_f1"
] = 1.0
mean_corr = full_corr["answer_f1"].mean()

print(f"{'Stima':38s} {'n':>6}  {'% su totale':>12}")
print("-" * 60)
print(f"{'Exact dopo norm (certo)':38s} {n_exact:>6}  {n_exact/n_total*100:>11.3f}%")
print(f"{'BLEURT>=0.7 (conservativa, ~85%)':38s} {n_cert:>6}  {n_cert/n_total*100:>11.3f}%")
print(f"{'BLEURT>=0.5 (stima ~27% precision)':38s} {est_mid:>6}  {est_mid/n_total*100:>11.3f}%")
print(f"{'BLEURT>=0.3 mid (upper bound)':38s} {n_mid:>6}  {n_mid/n_total*100:>11.3f}%")
print(f"\nAnswer F1 medio attuale:            {mean_now:.4f}")
print(f"Answer F1 medio post-correzione:    {mean_corr:.4f}  (Δ={mean_corr-mean_now:+.4f})")

# ── 5. Candidati falsi negativi ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. CANDIDATI FALSI NEGATIVI (F1=0)")
print("=" * 60)

zeros = full[full["answer_f1"] == 0].copy()
zeros["gold_norm"] = zeros["gold_answer"].astype(str).apply(normalize)
zeros["pred_norm"] = zeros["predicted_answer"].astype(str).apply(normalize)
zeros["gold_len"]  = zeros["gold_answer"].astype(str).str.len()

zeros["exact_norm"]   = zeros["gold_norm"] == zeros["pred_norm"]
zeros["gold_in_pred"] = zeros.apply(
    lambda r: r["gold_norm"] in r["pred_norm"] and r["gold_norm"] != r["pred_norm"],
    axis=1,
)
zeros["bleurt_high"] = zeros["bleurt_answer"] >= BLEURT_HIGH
zeros["bleurt_mid"]  = (
    zeros["bleurt_answer"].between(BLEURT_MID, BLEURT_HIGH, inclusive="left")
    & (zeros["gold_len"] <= SHORT_GOLD)
)

candidates = zeros[
    zeros["exact_norm"] | zeros["gold_in_pred"] | zeros["bleurt_high"] | zeros["bleurt_mid"]
].copy()


def label(row):
    tags = []
    if row["exact_norm"]:   tags.append("exact_norm")
    if row["gold_in_pred"]: tags.append("gold_in_pred")
    if row["bleurt_high"]:  tags.append("bleurt_high")
    if row["bleurt_mid"]:   tags.append("bleurt_mid")
    return "+".join(tags)


candidates["match_type"] = candidates.apply(label, axis=1)
candidates["correct"] = ""  # compila manualmente: y / n / ?

candidates = candidates.sort_values(
    ["exact_norm", "gold_in_pred", "bleurt_answer"],
    ascending=[False, False, False],
)

print(f"Totale F1=0:              {len(zeros):,}")
print(f"Candidati falsi negativi: {len(candidates):,} ({len(candidates)/len(zeros)*100:.1f}%)")
print(f"\nPer criterio (possono sovrapporsi):")
for col in ["exact_norm", "gold_in_pred", "bleurt_high", "bleurt_mid"]:
    print(f"  {col}: {candidates[col].sum()}")
print(f"\nEtichette combinate (top):")
print(candidates["match_type"].value_counts().head(8).to_string())

cols = [
    "match_type", "bleurt_answer",
    "question", "gold_answer", "predicted_answer", "answer_f1",
    "qid", "instruction_type", "group", "step", "run",
    "correct",
]
candidates[cols].to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
