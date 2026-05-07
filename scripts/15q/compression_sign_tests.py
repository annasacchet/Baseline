"""
Sign tests on compression ratio and BERTScore for 300q vs 15q.

Tests:
  1. Compression ratio step1 < 1.0 per instruction (il modello comprime invece di elaborare?)
  2. Compression ratio 300q < 15q step1 (il 4-bit comprime di più?)
  3. BERTScore baseline step1 < soglia (0.9) (drift significativo dall'originale?)
  4. Compression ratio decresce con le iterazioni step1 > step3 (compressione cumulativa?)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
CHAINS_15Q = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q.csv"
CHAINS_300Q = REPO_ROOT / "results" / "300q" / "rewriting_chains_300q.csv"
BS_300Q    = REPO_ROOT / "results" / "300q" / "rewriting_chains_300q_bertscore.csv"
OUT_CSV    = REPO_ROOT / "results" / "300q" / "compression_sign_tests.csv"

INSTRUCTIONS = ["elaborate", "shorten", "paraphrase", "formality"]
BERT_THRESHOLD = 0.9


def stars(p: float) -> str:
    if np.isnan(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def sign_test_vs_value(series: pd.Series, value: float, alternative: str = "less") -> dict:
    """H0: P(series < value) = 0.5."""
    below = int((series < value).sum())
    above = int((series > value).sum())
    n     = below + above
    if n == 0:
        return {"below": 0, "above": 0, "n": 0, "p": float("nan"), "mean": float("nan")}
    k = below if alternative == "less" else above
    result = stats.binomtest(k, n, 0.5, alternative="greater")
    return {"below": below, "above": above, "n": n, "p": result.pvalue, "mean": series.mean()}


def sign_test_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided sign test: H0 = P(a > b) = 0.5."""
    diff = a - b
    pos  = int((diff > 0).sum())
    neg  = int((diff < 0).sum())
    n    = pos + neg
    if n == 0:
        return {"pos": 0, "neg": 0, "n": 0, "p": float("nan"), "direction": "n/a"}
    result    = stats.binomtest(min(pos, neg), n, 0.5, alternative="two-sided")
    direction = "a>b" if pos > neg else "a<b" if neg > pos else "tie"
    return {"pos": pos, "neg": neg, "n": n, "p": result.pvalue, "direction": direction}


def compression_ratio(df: pd.DataFrame) -> pd.DataFrame:
    orig   = df[df['step'] == 0][['qid','instruction_type','run','n_tokens']].rename(columns={'n_tokens':'n_orig'})
    merged = df.merge(orig, on=['qid','instruction_type','run'])
    merged['compression_ratio'] = merged['n_tokens'] / merged['n_orig']
    return merged


def main():
    df300 = pd.read_csv(CHAINS_300Q)
    bs300 = pd.read_csv(BS_300Q)

    cr300 = compression_ratio(df300)

    rows = []

    # ------------------------------------------------------------------
    print("\n" + "#"*60)
    print("  TEST 1: Compression ratio step1 < 1.0")
    print("  (il modello comprime invece di elaborare?)")
    print("#"*60)

    for instr in INSTRUCTIONS:
        sub = cr300[(cr300['instruction_type'] == instr) & (cr300['step'] == 1)]['compression_ratio']
        r   = sign_test_vs_value(sub, 1.0, alternative="less")
        s   = stars(r["p"])
        print(f"  {instr:12s}: mean={r['mean']:.3f} | {r['below']}/{r['n']} < 1.0 | p={r['p']:.4f} {s}")
        rows.append({"test": "1_compression_vs_1.0", "dataset": "300q",
                     "subset": instr, "mean_a": round(r['mean'],4), "threshold": 1.0,
                     "n": r['n'], "pos": r['below'], "neg": r['above'],
                     "direction": "a<1.0", "p": round(r['p'],6), "sig": s})
    sub = cr300[cr300['step'] == 1]['compression_ratio']
    r   = sign_test_vs_value(sub, 1.0, alternative="less")
    s   = stars(r["p"])
    print(f"  {'TOTALE':12s}: mean={r['mean']:.3f} | {r['below']}/{r['n']} < 1.0 | p={r['p']:.4f} {s}")
    rows.append({"test": "1_compression_vs_1.0", "dataset": "300q",
                 "subset": "TOTALE", "mean_a": round(r['mean'],4), "threshold": 1.0,
                 "n": r['n'], "pos": r['below'], "neg": r['above'],
                 "direction": "a<1.0", "p": round(r['p'],6), "sig": s})


    # ------------------------------------------------------------------
    print("\n\n" + "#"*60)
    print(f"  TEST 3: BERTScore baseline step1 < {BERT_THRESHOLD}")
    print("  (drift significativo dall'originale?)")
    print("#"*60)

    for instr in INSTRUCTIONS:
        sub = bs300[(bs300['instruction_type'] == instr) & (bs300['step'] == 1)]['bert_f1_baseline']
        r   = sign_test_vs_value(sub, BERT_THRESHOLD, alternative="less")
        s   = stars(r["p"])
        print(f"  {instr:12s}: mean={r['mean']:.3f} | {r['below']}/{r['n']} < {BERT_THRESHOLD} | p={r['p']:.4f} {s}")
        rows.append({"test": f"3_bertscore_vs_{BERT_THRESHOLD}", "dataset": "300q",
                     "subset": instr, "mean_a": round(r['mean'],4), "threshold": BERT_THRESHOLD,
                     "n": r['n'], "pos": r['below'], "neg": r['above'],
                     "direction": f"a<{BERT_THRESHOLD}", "p": round(r['p'],6), "sig": s})

    sub = bs300[bs300['step'] == 1]['bert_f1_baseline']
    r   = sign_test_vs_value(sub, BERT_THRESHOLD, alternative="less")
    s   = stars(r["p"])
    print(f"  {'TOTALE':12s}: mean={r['mean']:.3f} | {r['below']}/{r['n']} < {BERT_THRESHOLD} | p={r['p']:.4f} {s}")
    rows.append({"test": f"3_bertscore_vs_{BERT_THRESHOLD}", "dataset": "300q",
                 "subset": "TOTALE", "mean_a": round(r['mean'],4), "threshold": BERT_THRESHOLD,
                 "n": r['n'], "pos": r['below'], "neg": r['above'],
                 "direction": f"a<{BERT_THRESHOLD}", "p": round(r['p'],6), "sig": s})

    # ------------------------------------------------------------------
    print("\n\n" + "#"*60)
    print("  TEST 3: Compression ratio step1 > step3")
    print("  (la compressione peggiora con le iterazioni?)")
    print("#"*60)

    for instr in INSTRUCTIONS:
        s1  = cr300[(cr300['instruction_type'] == instr) & (cr300['step'] == 1)].set_index(['qid','run'])['compression_ratio']
        s3  = cr300[(cr300['instruction_type'] == instr) & (cr300['step'] == 3)].set_index(['qid','run'])['compression_ratio']
        idx = s1.index.intersection(s3.index)
        r   = sign_test_paired(s1[idx].values, s3[idx].values)
        s   = stars(r["p"])
        print(f"  {instr:12s}: mean step1={s1.mean():.3f} → step3={s3.mean():.3f} | direction={r['direction']} | p={r['p']:.4f} {s}")
        rows.append({"test": "3_compression_step1_vs_step3", "dataset": "300q",
                     "subset": instr, "mean_a": round(s1.mean(),4), "mean_b": round(s3.mean(),4),
                     "n": r['n'], "pos": r['pos'], "neg": r['neg'],
                     "direction": r['direction'], "p": round(r['p'],6), "sig": s})

    # ------------------------------------------------------------------
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nRisultati salvati in: {OUT_CSV}")

    print("\n\n" + "#"*60)
    print("  CONCLUSIONI")
    print("#"*60)
    print(f"""
Test 1 — Compression ratio < 1.0
  Se significativo per tutte le instruction: il modello comprime
  sistematicamente invece di elaborare, indipendentemente dall'istruzione.

Test 2 — 300q (4-bit) vs 15q (bfloat16)
  Confronto descrittivo (campioni diversi). Se la media 300q è
  sostanzialmente inferiore alla 15q, la quantizzazione e/o il
  cambio di modello spiegano la differenza.

Test 3 — BERTScore < {BERT_THRESHOLD}
  Se significativo: il drift dall'originale è sistematico e non
  riconducibile a variabilità casuale.

Test 4 — Compression ratio step1 > step3
  Se significativo: la compressione peggiora iterazione dopo
  iterazione — ogni riscrittura perde ulteriore contenuto.

Legend: *** p<0.001  ** p<0.01  * p<0.05  ns not significant
""")


if __name__ == "__main__":
    main()
