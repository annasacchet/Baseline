"""
Significance tests for the four open investigations on the 15q pilot.

Tests:
  1. Late recovery non casuale (binomial + chi-square per istruzione)
  2. n_added correla con drop F1 (Spearman + Pearson) — elaborate
  3. n_added cresce step-by-step (Friedman + Wilcoxon paired) — elaborate
  4. Refusal/Wrong/Empty dipende da istruzione (chi-square)

Output:
  - results/15q/investigations_sign_tests.csv
  - stdout report
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
F1_CSV    = REPO_ROOT / "results/15q/rewriting_chains_15q_answer_f1.csv"
ELAB_CSV  = REPO_ROOT / "results/15q/elaborate_gpt_analysis.csv"
OUT_CSV   = REPO_ROOT / "results/15q/investigations_sign_tests.csv"


def classify_pred(pred):
    if pd.isna(pred) or str(pred).strip() == "" or str(pred).lower() == "nan":
        return "EMPTY"
    p = str(pred).lower()
    refusal_phrases = [
        "does not provide", "cannot answer", "not specified", "not provided",
        "no information", "cannot determine", "unable to", "context does not",
        "cannot be determined", "is not mentioned", "no specific", "don't know",
        "i cannot", "no clear", "unanswerable", "none provided", "none of the",
        "the question cannot",
    ]
    for r in refusal_phrases:
        if r in p:
            return "REFUSAL"
    return "WRONG"


def trajectory_cat(s1, s2, s3):
    if s2 == 0 and s3 == 0:
        return "STAYED_ZERO"
    if s3 > 0 and s2 == 0:
        return "LATE_RECOVERY"
    if s3 > 0 and s2 > 0:
        return "LATE_RECOVERY"
    if s2 > 0 and s3 == 0:
        return "TRANSIENT_RECOVERY"
    return "OTHER"


def main():
    df_f1 = pd.read_csv(F1_CSV)
    df_el = pd.read_csv(ELAB_CSV)

    rows = []  # for OUT_CSV summary

    print("=" * 78)
    print("TEST 1 — Late recovery non casuale")
    print("=" * 78)

    step1 = df_f1[df_f1["step"] == 1]
    zero_keys = list(zip(step1[step1["answer_f1"] == 0]["qid"],
                         step1[step1["answer_f1"] == 0]["instruction_type"],
                         step1[step1["answer_f1"] == 0]["run"]))
    n_zero = len(zero_keys)
    print(f"Chain con F1=0 a step 1: {n_zero}")

    cats = []
    for qid, instr, run in zero_keys:
        sub = df_f1[(df_f1["qid"] == qid)
                    & (df_f1["instruction_type"] == instr)
                    & (df_f1["run"] == run)].sort_values("step")
        f1s = sub.set_index("step")["answer_f1"].to_dict()
        s1, s2, s3 = f1s.get(1, 0), f1s.get(2, 0), f1s.get(3, 0)
        cats.append({"qid": qid, "instr": instr, "run": run,
                     "s1": s1, "s2": s2, "s3": s3,
                     "cat": trajectory_cat(s1, s2, s3)})
    cats_df = pd.DataFrame(cats)
    print("Trajectory counts:")
    print(cats_df["cat"].value_counts().to_string())

    # 1a: binomial test — H0: la prob. di recovery (s3>0) per chain F1=0 a step 1 è
    # uguale al rate di chain answerable nel resto del dataset.
    n_recovered_s3 = int((cats_df["s3"] > 0).sum())
    n_total_zero   = len(cats_df)
    # baseline rate from other 84 chains: tutte hanno F1>0 a step 1, quindi non
    # è la base giusta; usiamo invece la prob. che una chain answerable a step 1
    # sia ancora answerable a step 3 (per controllo "noise").
    answerable_keys = list(zip(step1[step1["answer_f1"] > 0]["qid"],
                                step1[step1["answer_f1"] > 0]["instruction_type"],
                                step1[step1["answer_f1"] > 0]["run"]))
    n_ans_s3 = 0
    for qid, instr, run in answerable_keys:
        s3 = df_f1[(df_f1["qid"] == qid)
                   & (df_f1["instruction_type"] == instr)
                   & (df_f1["run"] == run)
                   & (df_f1["step"] == 3)]["answer_f1"].iloc[0]
        if s3 > 0:
            n_ans_s3 += 1
    p_ans = n_ans_s3 / len(answerable_keys)
    print(f"\nProb. F1>0 a step 3 sulle 84 answerable: {p_ans:.3f}")
    print(f"Recovery a step 3 sulle 96 zero-at-s1: {n_recovered_s3}/{n_total_zero} "
          f"= {n_recovered_s3/n_total_zero:.3f}")

    # H0: rate di "F1>0 a step 3" è uguale tra zero-at-s1 e answerable-at-s1.
    # Two-proportion z-test (or Fisher's exact)
    contingency = np.array([
        [n_recovered_s3, n_total_zero - n_recovered_s3],
        [n_ans_s3,       len(answerable_keys) - n_ans_s3],
    ])
    odds, p_fisher = stats.fisher_exact(contingency, alternative="two-sided")
    print(f"Fisher's exact 2-prop (zero-at-s1 vs answerable-at-s1, F1>0 at s3):")
    print(f"  odds ratio = {odds:.4f}, p = {p_fisher:.3e}")

    # 1b: binomial test — H0: la "recovery rate" (8/96) è il rate attteso per puro
    # rumore. Non abbiamo un baseline preciso, ma usiamo p=0.05 come threshold
    # di "rumore casuale".
    res_bin = stats.binomtest(n_recovered_s3, n_total_zero, p=0.05,
                              alternative="greater")
    print(f"\nBinomial test (H0: recovery_rate ≤ 5%, alt: >):")
    print(f"  observed = {n_recovered_s3}/{n_total_zero}, p = {res_bin.pvalue:.3e}")

    # 1c: chi-square sulla distribuzione per istruzione delle chain con qualche
    # recovery (LATE+TRANSIENT). H0: distribuzione uniforme.
    rec_only = cats_df[cats_df["cat"] != "STAYED_ZERO"]
    counts_recovery = rec_only["instr"].value_counts().reindex(
        ["paraphrase", "shorten", "elaborate", "formality"]).fillna(0).astype(int)
    print(f"\nRecovery counts per instruction: {counts_recovery.to_dict()}")
    chi2, p_chi, dof, exp = stats.chi2_contingency([
        counts_recovery.values,
        [(counts_recovery.sum() / 4)] * 4,
    ])
    # better: chi-square goodness of fit con expected uniforme
    chi2_g, p_chi_g = stats.chisquare(counts_recovery.values)
    print(f"Chi-square goodness-of-fit (H0: uniforme su 4 istruzioni):")
    print(f"  chi2 = {chi2_g:.3f}, dof = 3, p = {p_chi_g:.3e}")

    rows.append(dict(test="1a_fisher_recovery_vs_answerable",
                     stat=odds, pvalue=p_fisher, n=n_zero + len(answerable_keys),
                     note=f"recovered {n_recovered_s3}/{n_total_zero} vs {n_ans_s3}/{len(answerable_keys)}"))
    rows.append(dict(test="1b_binomial_recovery_vs_5pct",
                     stat=n_recovered_s3 / n_total_zero, pvalue=res_bin.pvalue,
                     n=n_total_zero, note="alternative=greater, H0 p=0.05"))
    rows.append(dict(test="1c_chisq_uniform_per_instruction",
                     stat=chi2_g, pvalue=p_chi_g,
                     n=int(counts_recovery.sum()),
                     note=f"obs={counts_recovery.tolist()} (paraphrase, shorten, elaborate, formality)"))

    print("\n" + "=" * 78)
    print("TEST 2 — n_added correla con drop F1 (elaborate, comparison 0→1, 1→2, 2→3)")
    print("=" * 78)

    # Costruisce coppie (n_added, ΔF1) per le 45 chain elaborate × 3 confronti
    # consecutivi. ΔF1 = F1(tgt_step) - F1(src_step).
    el_pairs = df_el[df_el["comparison"].isin(["0→1", "1→2", "2→3"])].copy()
    f1_lookup = df_f1.set_index(["qid", "instruction_type", "run", "step"])["answer_f1"].to_dict()

    deltas = []
    n_added_vals = []
    for _, r in el_pairs.iterrows():
        key_src = (r["qid"], "elaborate", int(r["run"]), int(r["src_step"]))
        key_tgt = (r["qid"], "elaborate", int(r["run"]), int(r["tgt_step"]))
        f1_src = f1_lookup.get(key_src)
        f1_tgt = f1_lookup.get(key_tgt)
        if f1_src is None or f1_tgt is None:
            continue
        deltas.append(f1_tgt - f1_src)
        n_added_vals.append(r["n_added"])

    deltas = np.array(deltas, dtype=float)
    n_added_vals = np.array(n_added_vals, dtype=float)
    print(f"n coppie: {len(deltas)}")
    rho, p_rho = stats.spearmanr(n_added_vals, deltas)
    r_p, p_r   = stats.pearsonr(n_added_vals, deltas)
    print(f"Spearman ρ = {rho:.3f}, p = {p_rho:.3e}")
    print(f"Pearson  r = {r_p:.3f}, p = {p_r:.3e}")
    rows.append(dict(test="2a_spearman_nadded_vs_deltaF1",
                     stat=rho, pvalue=p_rho, n=len(deltas),
                     note="elaborate, consecutive comparisons"))
    rows.append(dict(test="2b_pearson_nadded_vs_deltaF1",
                     stat=r_p, pvalue=p_r, n=len(deltas),
                     note="elaborate, consecutive comparisons"))

    # subset: solo chain answerable a step 1 (per evitare floor effect)
    answerable_set = set(answerable_keys)
    deltas_a = []
    nadd_a = []
    for _, r in el_pairs.iterrows():
        chain_key = (r["qid"], "elaborate", int(r["run"]))
        if chain_key not in answerable_set:
            continue
        key_src = (r["qid"], "elaborate", int(r["run"]), int(r["src_step"]))
        key_tgt = (r["qid"], "elaborate", int(r["run"]), int(r["tgt_step"]))
        f1_src = f1_lookup.get(key_src)
        f1_tgt = f1_lookup.get(key_tgt)
        if f1_src is None or f1_tgt is None:
            continue
        deltas_a.append(f1_tgt - f1_src)
        nadd_a.append(r["n_added"])
    print(f"\n[subset answerable-at-s1] n coppie: {len(deltas_a)}")
    if len(deltas_a) >= 5:
        rho_a, p_rho_a = stats.spearmanr(nadd_a, deltas_a)
        r_a, p_r_a = stats.pearsonr(nadd_a, deltas_a)
        print(f"Spearman ρ = {rho_a:.3f}, p = {p_rho_a:.3e}")
        print(f"Pearson  r = {r_a:.3f}, p = {p_r_a:.3e}")
        rows.append(dict(test="2c_spearman_nadded_vs_deltaF1_answerable",
                         stat=rho_a, pvalue=p_rho_a, n=len(deltas_a),
                         note="elaborate, consec comparisons, subset chain answerable a step 1"))

    print("\n" + "=" * 78)
    print("TEST 3 — n_added cresce step-by-step (elaborate)")
    print("=" * 78)

    pivot = df_el[df_el["comparison"].isin(["0→1", "1→2", "2→3"])].pivot_table(
        index=["qid", "run"], columns="comparison", values="n_added"
    )
    print("Means per comparison:")
    print(pivot.mean().round(3).to_string())
    print(f"\nN chain (qid, run) complete: {pivot.dropna().shape[0]}")

    # Friedman test on the three repeated measures (n_added at 0→1, 1→2, 2→3)
    pivot_clean = pivot.dropna()
    if pivot_clean.shape[0] >= 3:
        fr_stat, fr_p = stats.friedmanchisquare(
            pivot_clean["0→1"], pivot_clean["1→2"], pivot_clean["2→3"]
        )
        print(f"Friedman: chi2 = {fr_stat:.3f}, p = {fr_p:.3e}")
        rows.append(dict(test="3a_friedman_nadded_steps",
                         stat=fr_stat, pvalue=fr_p, n=pivot_clean.shape[0],
                         note="repeated measures n_added at 0→1, 1→2, 2→3"))

        # Wilcoxon paired, due confronti
        w1, p_w1 = stats.wilcoxon(pivot_clean["0→1"], pivot_clean["1→2"])
        w2, p_w2 = stats.wilcoxon(pivot_clean["1→2"], pivot_clean["2→3"])
        print(f"Wilcoxon 0→1 vs 1→2: W = {w1:.3f}, p = {p_w1:.3e}")
        print(f"Wilcoxon 1→2 vs 2→3: W = {w2:.3f}, p = {p_w2:.3e}")
        rows.append(dict(test="3b_wilcoxon_nadded_01_vs_12",
                         stat=w1, pvalue=p_w1, n=pivot_clean.shape[0]))
        rows.append(dict(test="3c_wilcoxon_nadded_12_vs_23",
                         stat=w2, pvalue=p_w2, n=pivot_clean.shape[0]))

    print("\n" + "=" * 78)
    print("TEST 4 — Refusal/Wrong/Empty dipende da istruzione (chi-square)")
    print("=" * 78)

    zero_step1 = step1[step1["answer_f1"] == 0].copy()
    zero_step1["category"] = zero_step1["predicted_answer"].apply(classify_pred)
    crosstab = pd.crosstab(zero_step1["instruction_type"], zero_step1["category"])
    print("Contingency table:")
    print(crosstab.to_string())

    chi2, p_chi, dof, expected = stats.chi2_contingency(crosstab.values)
    print(f"\nChi-square: chi2 = {chi2:.3f}, dof = {dof}, p = {p_chi:.3e}")
    # Cramer's V
    n_total = crosstab.values.sum()
    cramer_v = np.sqrt(chi2 / (n_total * (min(crosstab.shape) - 1)))
    print(f"Cramer's V = {cramer_v:.3f}")
    rows.append(dict(test="4a_chisq_category_vs_instruction",
                     stat=chi2, pvalue=p_chi, n=int(n_total),
                     note=f"dof={dof}; cramer_v={cramer_v:.3f}; cats={crosstab.columns.tolist()}"))

    # If expected counts < 5, also report Fisher's exact (m×n via simulation)
    low_expected = (expected < 5).sum()
    if low_expected > 0:
        print(f"WARNING: {low_expected} celle con expected count < 5; chi-square può essere instabile.")
        # Fisher's exact con permutation (Monte Carlo)
        try:
            res_perm = stats.chi2_contingency(crosstab.values, lambda_="log-likelihood")
            print(f"G-test (likelihood ratio): chi2 = {res_perm[0]:.3f}, p = {res_perm[1]:.3e}")
            rows.append(dict(test="4b_g_test_category_vs_instruction",
                             stat=res_perm[0], pvalue=res_perm[1], n=int(n_total),
                             note="log-likelihood G-test (più robusto a low expected)"))
        except Exception as e:
            print(f"G-test failed: {e}")

    # Save summary CSV
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
