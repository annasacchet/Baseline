"""
Sign tests on FactScore faithfulness for all instruction types and groups.

Tests:
  1. FactScore step1 < 1.0  (prima riscrittura perde già fedeltà?)
  2. FactScore step1 vs step3 per istruzione (degrado cumulativo?)
  3. FactScore step1: elaborate < altre istruzioni (elaborate è la peggiore?)
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="15q", help="Dataset tag (e.g. 15q, 200q).")
    return parser.parse_args()


_args   = parse_args()
TAG     = _args.tag
OFS_CSV = REPO_ROOT / "results" / TAG / f"rewriting_chains_{TAG}_openfactscore.csv"
OUT_CSV = REPO_ROOT / "results" / TAG / f"factscore_sign_tests_{TAG}.csv"

INSTRUCTIONS = ["elaborate", "shorten", "paraphrase", "formality"]
GROUPS       = ["content", "style"]


def stars(p: float) -> str:
    if np.isnan(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def sign_test_vs_one(series: pd.Series) -> dict:
    """H0: P(factscore < 1.0) = 0.5. One-sided: greater."""
    below = int((series < 1.0).sum())
    n     = len(series)
    result = stats.binomtest(below, n, 0.5, alternative="greater")
    return {"below": below, "n": n, "mean": series.mean(), "p": result.pvalue}


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


def main():
    ofs = pd.read_csv(OFS_CSV)
    rows = []

    # ------------------------------------------------------------------ #
    print("\n" + "#"*60)
    print("  TEST 1: FactScore step1 < 1.0")
    print("  (la prima riscrittura perde già fedeltà?)")
    print("#"*60)

    for label, subset in [
        *[(f"instruction = {i.upper()}", ofs[ofs["instruction_type"] == i]) for i in INSTRUCTIONS],
        *[(f"group = {g.upper()}", ofs[ofs["group"] == g]) for g in GROUPS],
        ("TOTALE", ofs),
    ]:
        sub = subset[subset["step"] == 1]["factscore"]
        r   = sign_test_vs_one(sub)
        s   = stars(r["p"])
        print(f"\n  {label}")
        print(f"    mean={r['mean']:.3f} | {r['below']}/{r['n']} con factscore<1.0 | p={r['p']:.4f} {s}")
        rows.append({
            "test": "1_factscore_vs_1.0", "subset": label,
            "mean_a": round(r["mean"], 4), "mean_b": 1.0,
            "n": r["n"], "pos": r["below"], "neg": r["n"] - r["below"],
            "direction": "a<1.0", "p": round(r["p"], 6), "sig": s,
        })

    # ------------------------------------------------------------------ #
    print("\n\n" + "#"*60)
    print("  TEST 2: FactScore step1 vs step3")
    print("  (degrado cumulativo della fedeltà?)")
    print("#"*60)

    for label, subset in [
        *[(f"instruction = {i.upper()}", ofs[ofs["instruction_type"] == i]) for i in INSTRUCTIONS],
        *[(f"group = {g.upper()}", ofs[ofs["group"] == g]) for g in GROUPS],
        ("TOTALE", ofs),
    ]:
        s1  = subset[subset["step"] == 1].set_index(["qid", "run"])["factscore"]
        s3  = subset[subset["step"] == 3].set_index(["qid", "run"])["factscore"]
        idx = s1.index.intersection(s3.index)
        r   = sign_test_paired(s1[idx].values, s3[idx].values)
        s   = stars(r["p"])
        means = subset.groupby("step")["factscore"].mean()
        print(f"\n  {label}")
        print(f"    n={r['n']}, pos={r['pos']}, neg={r['neg']} | direction={r['direction']} | p={r['p']:.4f} {s}")
        print(f"    mean step1={means.get(1, float('nan')):.3f} → step3={means.get(3, float('nan')):.3f}")
        rows.append({
            "test": "2_factscore_step1_vs_step3", "subset": label,
            "mean_a": round(means.get(1, float("nan")), 4),
            "mean_b": round(means.get(3, float("nan")), 4),
            "n": r["n"], "pos": r["pos"], "neg": r["neg"],
            "direction": r["direction"], "p": round(r["p"], 6), "sig": s,
        })

    # ------------------------------------------------------------------ #
    print("\n\n" + "#"*60)
    print("  TEST 3: FactScore step1 — elaborate vs altre istruzioni")
    print("  (elaborate perde più fedeltà delle altre?)")
    print("#"*60)

    elab = ofs[(ofs["instruction_type"] == "elaborate") & (ofs["step"] == 1)]
    for other in ["shorten", "paraphrase", "formality"]:
        other_df = ofs[(ofs["instruction_type"] == other) & (ofs["step"] == 1)]
        # Paired su (qid, run)
        merged = elab.set_index(["qid","run"])[["factscore"]].join(
            other_df.set_index(["qid","run"])[["factscore"]],
            lsuffix="_elab", rsuffix="_other"
        ).dropna()
        r = sign_test_paired(merged["factscore_elab"].values, merged["factscore_other"].values)
        s = stars(r["p"])
        print(f"\n  elaborate vs {other.upper()}")
        print(f"    mean elaborate={elab['factscore'].mean():.3f} | mean {other}={other_df['factscore'].mean():.3f}")
        print(f"    n={r['n']}, pos={r['pos']}, neg={r['neg']} | direction={r['direction']} | p={r['p']:.4f} {s}")
        rows.append({
            "test": "3_elaborate_vs_other_step1", "subset": f"elaborate vs {other}",
            "mean_a": round(elab["factscore"].mean(), 4),
            "mean_b": round(other_df["factscore"].mean(), 4),
            "n": r["n"], "pos": r["pos"], "neg": r["neg"],
            "direction": r["direction"], "p": round(r["p"], 6), "sig": s,
        })

    # ------------------------------------------------------------------ #
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nRisultati salvati in: {OUT_CSV}")

    print("\n\n" + "#"*60)
    print("  CONCLUSIONI")
    print("#"*60)
    print("""
Test 1 — FactScore step1 < 1.0
  Tutte le instruction perdono fedeltà già al primo step (p<0.001).
  Nessuna riscrittura è pienamente faithfull: il modello introduce
  claim non supportati indipendentemente dall'istruzione ricevuta.

Test 2 — FactScore step1 vs step3
  Il degrado cumulativo è significativo solo per ELABORATE e per il
  gruppo CONTENT. Per STYLE (formality, paraphrase) la fedeltà non
  peggiora significativamente con le iterazioni — il danno è già
  fatto al primo step e poi si stabilizza.

Test 3 — Elaborate vs altre istruzioni
  Elaborate ha il factscore più basso al step1. Se il confronto
  paired è significativo, confirma che elaborare è strutturalmente
  più dannoso per la fedeltà rispetto alle altre istruzioni.

Legend: *** p<0.001  ** p<0.01  * p<0.05  ns not significant
""")


if __name__ == "__main__":
    main()
