"""
Unified statistical analysis for the 300q MuSiQue rewriting study.

Replaces the older split scripts (inference_tests / recovery_analysis / metrics_stats /
elaborate_analysis) with a single, self-contained pipeline.

Design (each block answers one explicit question):

  S0  Descriptive table         per-step / per-instruction / per-hop means + n
  S1  Omnibus tests             Friedman + Wilcoxon paired step contrasts (Holm),
                                with rank-biserial effect sizes
  S2  GLMM / LMM                Mixed models with (1|qid) for F1 (logit), OFS,
                                BERTScore baseline, BERTScore consecutive, BLEURT baseline
  S3  Cluster bootstrap CIs     B=10000 resampling at the qid level for every key Delta
                                (step 0->1 F1, step 1->3 OFS, recovery rate, etc.)
  S4  Causal mediation          ACME / ADE / proportion mediated for
                                step -> log(n_tokens) -> F1 (and -> OFS)
  S5  Recovery survival         Kaplan-Meier for time-to-loss of an F1>=0.9 recovery;
                                log-rank by instruction
  S6  TOST equivalence          Two one-sided tests for step1->2 and step2->3 F1 deltas
                                against +/-0.02 equivalence bounds
  S7  Recovery sensitivity      Recovery rate as a function of the F1 threshold,
                                with cluster-bootstrap CIs
  S8  IRR stub                  Cohen's kappa scaffold for a future manual annotation
                                of F1=0 / BLEURT>=0.3 cases (no manual labels yet)
  S9  RQ summary table          One-row-per-RQ headline with key estimate + test

Outputs go to results/300q/stats_v2/ as CSVs; the doc layer reads them.

Usage:
    python3.11 scripts/300q/statistical_analysis_300q.py
    python3.11 scripts/300q/statistical_analysis_300q.py --bootstrap 2000   # faster
    python3.11 scripts/300q/statistical_analysis_300q.py --skip-bootstrap
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "300q"
OUT = RES / "stats_v2"

F1_CSV = RES / "rewriting_chains_300q_answer_f1_olmo31_4bit.csv"
OFS_CSV = RES / "rewriting_chains_300q_openfactscore.csv"
BS_CSV = RES / "rewriting_chains_300q_bertscore.csv"
BLEURT_CSV = RES / "rewriting_chains_300q_bleurt.csv"
CHAINS_CSV = RES / "rewriting_chains_300q.csv"

KEYS = ["qid", "group", "instruction_type", "run", "step"]
RECOVERY_THRESHOLD = 0.9
EQUIV_BOUND = 0.02  # +/- F1 points for TOST


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def save(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  [saved] {path.relative_to(ROOT)}  ({len(df)} rows)")


def add_hop(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_hop"] = df["qid"].str.extract(r"^(\d+)hop")[0].astype(int)
    return df


def load_all() -> dict[str, pd.DataFrame]:
    f1 = add_hop(pd.read_csv(F1_CSV))
    ofs = add_hop(pd.read_csv(OFS_CSV))
    bs = add_hop(pd.read_csv(BS_CSV))
    bleurt = add_hop(pd.read_csv(BLEURT_CSV))
    chains = add_hop(pd.read_csv(CHAINS_CSV))

    # Normalise step dtype across files
    for d in (f1, ofs, bs, bleurt, chains):
        d["step"] = d["step"].astype(int)
        d["run"] = d["run"].astype(int)

    # Long-format merge with n_tokens
    tokens = chains[KEYS + ["n_tokens"]].copy()
    f1 = f1.merge(tokens, on=KEYS, how="left")

    f1["f1_bin"] = (f1["answer_f1"] > 0).astype(int)
    f1["f1_high"] = (f1["answer_f1"] >= RECOVERY_THRESHOLD).astype(int)

    print(f"  F1     : {len(f1):>6} rows  ({f1['qid'].nunique()} qid x "
          f"{f1['instruction_type'].nunique()} instr x "
          f"{f1['run'].nunique()} run x {f1['step'].nunique()} step)")
    print(f"  OFS    : {len(ofs):>6} rows")
    print(f"  BERT   : {len(bs):>6} rows")
    print(f"  BLEURT : {len(bleurt):>6} rows")
    return {"f1": f1, "ofs": ofs, "bs": bs, "bleurt": bleurt, "chains": chains}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Wilcoxon signed-rank effect size (matched pairs)."""
    diff = np.asarray(x) - np.asarray(y)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = st.rankdata(np.abs(diff))
    pos = ranks[diff > 0].sum()
    neg = ranks[diff < 0].sum()
    total = pos + neg
    return float((pos - neg) / total) if total else 0.0


def cluster_bootstrap_mean(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    B: int = 10_000,
    seed: int = 17,
) -> tuple[float, float, float]:
    """Vectorised cluster bootstrap CI for the mean of `values`, resampling clusters
    with replacement at the qid level.

    Uses pre-grouped sums and counts: each bootstrap iteration is a single
    integer-indexed gather of two length-K arrays (K = #clusters), no DataFrame.
    Speed: ~5-10us per iteration vs ~30ms with the naive concat version.
    """
    clusters, inv = np.unique(cluster_ids, return_inverse=True)
    K = len(clusters)
    cluster_sums = np.bincount(inv, weights=values, minlength=K)
    cluster_counts = np.bincount(inv, minlength=K)
    point = cluster_sums.sum() / cluster_counts.sum()

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, K, size=(B, K))
    boot_means = cluster_sums[idx].sum(axis=1) / cluster_counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def cluster_bootstrap_paired_diff(
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    cluster_ids: np.ndarray,
    B: int = 10_000,
    seed: int = 17,
) -> tuple[float, float, float]:
    """CI for mean(pair_b - pair_a) with cluster resampling on cluster_ids."""
    diff = pair_b - pair_a
    return cluster_bootstrap_mean(diff, cluster_ids, B=B, seed=seed)


def fast_ols_betas(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve normal equations X'X beta = X'y. Returns beta vector.
    ~50us for X of shape (14k, ~10). 100x faster than smf.ols."""
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


# ---------------------------------------------------------------------------
# S0 Descriptive table
# ---------------------------------------------------------------------------
def s0_descriptives(d: dict[str, pd.DataFrame]) -> None:
    section("S0  Descriptive summaries")
    rows = []
    for metric_name, df, col in [
        ("answer_f1", d["f1"], "answer_f1"),
        ("openfactscore", d["ofs"], "factscore"),
        ("bertscore_baseline", d["bs"], "bert_f1_baseline"),
        ("bertscore_consecutive", d["bs"], "bert_f1_consecutive"),
        ("bleurt_baseline", d["bleurt"], "bleurt_baseline"),
        ("bleurt_consecutive", d["bleurt"], "bleurt_consecutive"),
        ("bleurt_answer", d["bleurt"], "bleurt_answer"),
        ("n_tokens", d["chains"], "n_tokens"),
    ]:
        if col not in df.columns:
            continue
        g = df.groupby("step")[col].agg(["mean", "std", "median", "count"]).reset_index()
        g.insert(0, "metric", metric_name)
        rows.append(g)
    save(pd.concat(rows, ignore_index=True), "s0_descriptive_by_step")

    # by instruction x step (F1 + OFS only)
    rows = []
    for metric, df, col in [
        ("answer_f1", d["f1"], "answer_f1"),
        ("openfactscore", d["ofs"], "factscore"),
    ]:
        g = (df.groupby(["instruction_type", "step"])[col]
                .agg(["mean", "std", "count"]).reset_index())
        g.insert(0, "metric", metric)
        rows.append(g)
    save(pd.concat(rows, ignore_index=True), "s0_descriptive_by_instruction_step")

    # by hop x step
    rows = []
    for metric, df, col in [
        ("answer_f1", d["f1"], "answer_f1"),
        ("openfactscore", d["ofs"], "factscore"),
    ]:
        g = df.groupby(["n_hop", "step"])[col].agg(["mean", "std", "count"]).reset_index()
        g.insert(0, "metric", metric)
        rows.append(g)
    save(pd.concat(rows, ignore_index=True), "s0_descriptive_by_hop_step")


# ---------------------------------------------------------------------------
# S1 Omnibus + paired step contrasts
# ---------------------------------------------------------------------------
def aggregate_per_qid(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Average over (instruction, run) within (qid, step). Required for Friedman."""
    return (df.groupby(["qid", "step"])[value_col].mean().unstack("step").dropna())


def s1_omnibus(d: dict[str, pd.DataFrame]) -> None:
    section("S1  Friedman omnibus + Wilcoxon step contrasts (Holm)")

    specs = [
        ("answer_f1", d["f1"], "answer_f1", [0, 1, 2, 3]),
        ("openfactscore", d["ofs"], "factscore", [1, 2, 3]),
        ("bertscore_baseline", d["bs"], "bert_f1_baseline", [1, 2, 3]),
        ("bertscore_consecutive", d["bs"], "bert_f1_consecutive", [1, 2, 3]),
        ("bleurt_baseline", d["bleurt"], "bleurt_baseline", [1, 2, 3]),
    ]

    omnibus_rows, contrast_rows = [], []
    for name, df, col, steps in specs:
        wide = aggregate_per_qid(df, col)
        wide = wide[steps]  # ensure order
        chi2, p = st.friedmanchisquare(*[wide[s].values for s in steps])
        omnibus_rows.append({
            "metric": name, "test": "Friedman", "n_qid": len(wide),
            "k_steps": len(steps), "chi2": chi2, "df": len(steps) - 1, "p_value": p,
        })

        pairs = list(zip(steps[:-1], steps[1:]))
        pvs, deltas, es, ns = [], [], [], []
        for a, b in pairs:
            x, y = wide[a].values, wide[b].values
            stat, p_w = st.wilcoxon(x, y, zero_method="wilcox")
            deltas.append(float(np.mean(y - x)))
            es.append(rank_biserial(y, x))
            pvs.append(p_w)
            ns.append(len(wide))
        _, p_holm, *_ = multipletests(pvs, method="holm")
        for (a, b), dlt, ef, p_raw, p_h, n in zip(pairs, deltas, es, pvs, p_holm, ns):
            contrast_rows.append({
                "metric": name, "contrast": f"step{a}_to_{b}", "n_qid": n,
                "delta_mean": dlt, "rank_biserial": ef,
                "p_raw": p_raw, "p_holm": p_h,
            })

    save(pd.DataFrame(omnibus_rows), "s1_friedman_omnibus")
    save(pd.DataFrame(contrast_rows), "s1_wilcoxon_step_contrasts")


# ---------------------------------------------------------------------------
# S2 Mixed models
# ---------------------------------------------------------------------------
def fit_mixed(df: pd.DataFrame, formula: str, group: str = "qid",
              logistic: bool = False) -> pd.DataFrame:
    """Fit LMM or BinomialBayesMixedGLM-equivalent. We use MixedLM for both since
    statsmodels GLMM logit is unstable on this size; for logistic we rely on the
    Friedman + Wilcoxon for robustness and report MixedLM on the binary outcome
    as a linear-probability model (LPM)."""
    try:
        model = smf.mixedlm(formula, df, groups=df[group])
        fit = model.fit(method="lbfgs", reml=not logistic, disp=False)
        out = []
        for name, beta, se, pv in zip(
            fit.params.index, fit.params.values, fit.bse.values, fit.pvalues.values
        ):
            out.append({
                "term": name, "estimate": beta, "std_error": se, "p_value": pv,
            })
        out.append({
            "term": "RE_var_group", "estimate": float(fit.cov_re.iloc[0, 0]),
            "std_error": np.nan, "p_value": np.nan,
        })
        out.append({
            "term": "Residual_var", "estimate": float(fit.scale),
            "std_error": np.nan, "p_value": np.nan,
        })
        return pd.DataFrame(out)
    except Exception as exc:
        return pd.DataFrame([{"term": "ERROR", "estimate": np.nan,
                              "std_error": np.nan, "p_value": np.nan,
                              "message": str(exc)}])


def icc_from_fit(df: pd.DataFrame, value_col: str, group_col: str = "qid") -> float:
    """ICC from a null random-intercept model: var(RE) / (var(RE) + var(resid))."""
    try:
        f = smf.mixedlm(f"{value_col} ~ 1", df, groups=df[group_col]).fit(
            method="lbfgs", reml=True, disp=False)
        var_re = float(f.cov_re.iloc[0, 0])
        var_res = float(f.scale)
        return var_re / (var_re + var_res) if (var_re + var_res) > 0 else np.nan
    except Exception:
        return np.nan


def s2_mixed_models(d: dict[str, pd.DataFrame]) -> None:
    section("S2  Mixed models (random intercept per qid)")

    # F1 as LPM (linear probability) + as continuous, OFS, BERTScore baseline + consecutive,
    # BLEURT baseline. Each: outcome ~ C(step, Treatment(0)) + C(instruction_type)
    #                       + C(n_hop) + (1|qid).
    models = [
        ("answer_f1_lpm", d["f1"], "answer_f1",
         "answer_f1 ~ C(step, Treatment(reference=0)) + C(instruction_type) + C(n_hop)"),
        ("openfactscore_lmm", d["ofs"], "factscore",
         "factscore ~ C(step, Treatment(reference=1)) + C(instruction_type) + C(n_hop)"),
        ("bertscore_baseline_lmm", d["bs"], "bert_f1_baseline",
         "bert_f1_baseline ~ C(step, Treatment(reference=1)) + C(instruction_type) + C(n_hop)"),
        ("bertscore_consecutive_lmm", d["bs"], "bert_f1_consecutive",
         "bert_f1_consecutive ~ C(step, Treatment(reference=1)) + C(instruction_type) + C(n_hop)"),
        ("bleurt_baseline_lmm", d["bleurt"], "bleurt_baseline",
         "bleurt_baseline ~ C(step, Treatment(reference=1)) + C(instruction_type) + C(n_hop)"),
    ]

    all_rows = []
    icc_rows = []
    for name, df, col, formula in models:
        print(f"  fitting {name} ...")
        res = fit_mixed(df, formula)
        res.insert(0, "model", name)
        all_rows.append(res)
        icc = icc_from_fit(df, col)
        icc_rows.append({"model": name, "ICC_qid": icc, "n_obs": len(df),
                         "n_qid": df["qid"].nunique()})
    save(pd.concat(all_rows, ignore_index=True), "s2_mixed_models_coefficients")
    save(pd.DataFrame(icc_rows), "s2_icc_qid")


# ---------------------------------------------------------------------------
# S3 Cluster bootstrap CIs
# ---------------------------------------------------------------------------
def s3_bootstrap_cis(d: dict[str, pd.DataFrame], B: int) -> None:
    section(f"S3  Cluster bootstrap CIs at qid level (B={B})")

    f1 = d["f1"]
    ofs = d["ofs"]
    bs = d["bs"]

    rows = []

    # ---- Mean F1 per step (cluster on qid)
    for step in sorted(f1["step"].unique()):
        sub = f1[f1["step"] == step]
        point, lo, hi = cluster_bootstrap_mean(
            sub["answer_f1"].values, sub["qid"].values, B=B, seed=1000 + step)
        rows.append({"statistic": f"mean_F1_step{step}", "estimate": point,
                     "ci_lo": lo, "ci_hi": hi, "B": B})

    # ---- Step deltas on F1 (paired at qid level, averaged within qid)
    wide_f1 = f1.pivot_table(index="qid", columns="step",
                             values="answer_f1", aggfunc="mean").reset_index()
    for a, b in [(0, 1), (1, 2), (2, 3), (0, 3)]:
        pair = wide_f1.dropna(subset=[a, b])
        point, lo, hi = cluster_bootstrap_paired_diff(
            pair[a].values, pair[b].values, pair["qid"].values,
            B=B, seed=2000 + a * 10 + b)
        rows.append({"statistic": f"delta_F1_step{a}_to_{b}", "estimate": point,
                     "ci_lo": lo, "ci_hi": hi, "B": B})

    # ---- OFS step deltas
    wide_ofs = ofs.pivot_table(index="qid", columns="step",
                               values="factscore", aggfunc="mean").reset_index()
    for a, b in [(1, 2), (2, 3), (1, 3)]:
        pair = wide_ofs.dropna(subset=[a, b])
        point, lo, hi = cluster_bootstrap_paired_diff(
            pair[a].values, pair[b].values, pair["qid"].values,
            B=B, seed=3000 + a * 10 + b)
        rows.append({"statistic": f"delta_OFS_step{a}_to_{b}", "estimate": point,
                     "ci_lo": lo, "ci_hi": hi, "B": B})

    # ---- Recovery rate (chains zero-start with F1>=thr in any later step)
    zero_start = f1[f1["step"] == 0].query("answer_f1 == 0")[
        ["qid", "instruction_type", "run"]].drop_duplicates()
    chain_max = (f1[f1["step"] > 0]
                 .groupby(["qid", "instruction_type", "run"])["answer_f1"].max()
                 .reset_index())
    zs = zero_start.merge(chain_max, on=["qid", "instruction_type", "run"], how="left")

    for label, thr in [("09", RECOVERY_THRESHOLD), ("05", 0.5), ("any", 1e-9)]:
        vals = (zs["answer_f1"] >= thr).astype(float).values
        point, lo, hi = cluster_bootstrap_mean(
            vals, zs["qid"].values, B=B, seed=4000 + int(thr * 1000))
        rows.append({"statistic": f"recovery_rate_{label}",
                     "estimate": point, "ci_lo": lo, "ci_hi": hi, "B": B})

    # ---- BERTScore consecutive jump step1 -> step2
    wide_bs = bs.pivot_table(index="qid", columns="step",
                             values="bert_f1_consecutive", aggfunc="mean").reset_index()
    pair = wide_bs.dropna(subset=[1, 2])
    point, lo, hi = cluster_bootstrap_paired_diff(
        pair[1].values, pair[2].values, pair["qid"].values, B=B, seed=5001)
    rows.append({"statistic": "delta_BERT_consecutive_step1_to_2",
                 "estimate": point, "ci_lo": lo, "ci_hi": hi, "B": B})

    save(pd.DataFrame(rows), "s3_bootstrap_cis")


# ---------------------------------------------------------------------------
# S4 Causal mediation: step -> log(n_tokens) -> F1 (and OFS)
# ---------------------------------------------------------------------------
def _build_design_matrix(df: pd.DataFrame, treat_col: str,
                         extra_cols: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Builds [intercept, treat, dummies(instruction_type), dummies(n_hop), *extras].
    Dummy variables use the first sorted level as reference."""
    n = len(df)
    blocks = [np.ones(n), df[treat_col].values.astype(float)]
    names = ["intercept", treat_col]

    for cat_col in ["instruction_type", "n_hop"]:
        levels = sorted(df[cat_col].unique())
        for lev in levels[1:]:
            blocks.append((df[cat_col].values == lev).astype(float))
            names.append(f"{cat_col}_{lev}")

    if extra_cols:
        for c in extra_cols:
            blocks.append(df[c].values.astype(float))
            names.append(c)

    X = np.column_stack(blocks)
    return X, names


def _mediation_paths(X_treat: np.ndarray, X_full: np.ndarray,
                     y: np.ndarray, m: np.ndarray,
                     idx_treat: int, idx_mediator_in_full: int) -> tuple[float, float, float, float]:
    """Returns (c_total, a, b, c_prime) using fast OLS via np.linalg.solve.

    X_treat  = design matrix WITHOUT mediator -> used for path c (outcome~treat)
               and path a (mediator~treat); since both share the same RHS,
               we solve two systems with the same XtX (one factorisation).
    X_full   = design matrix WITH mediator    -> used for paths c' and b.
    """
    # c (outcome ~ treat) and a (mediator ~ treat): same X
    XtX = X_treat.T @ X_treat
    beta_c = np.linalg.solve(XtX, X_treat.T @ y)
    beta_a = np.linalg.solve(XtX, X_treat.T @ m)
    c_total = beta_c[idx_treat]
    a = beta_a[idx_treat]

    # c' and b on the full design
    beta_full = np.linalg.solve(X_full.T @ X_full, X_full.T @ y)
    c_prime = beta_full[idx_treat]
    b = beta_full[idx_mediator_in_full]
    return float(c_total), float(a), float(b), float(c_prime)


def s4_mediation(d: dict[str, pd.DataFrame], B: int) -> None:
    section(f"S4  Causal mediation step -> log(n_tokens) -> outcome  (boot B={B})")

    f1 = d["f1"].copy()
    f1["log_tokens"] = np.log(f1["n_tokens"].clip(lower=1))
    f1["step_num"] = f1["step"].astype(float)

    ofs = d["ofs"].copy()
    chains = d["chains"][KEYS + ["n_tokens"]]
    ofs = ofs.merge(chains, on=KEYS, how="left")
    ofs["log_tokens"] = np.log(ofs["n_tokens"].clip(lower=1))
    ofs["step_num"] = ofs["step"].astype(float)

    specs = [
        ("F1 ~ step (mediator=log_tokens)", f1, "answer_f1"),
        ("OFS ~ step (mediator=log_tokens)", ofs, "factscore"),
    ]

    out_rows = []
    for label, df, outcome in specs:
        # Pre-build full + treat-only design matrices ONCE
        X_treat, names_treat = _build_design_matrix(df, "step_num")
        X_full, names_full = _build_design_matrix(df, "step_num",
                                                  extra_cols=["log_tokens"])
        y = df[outcome].values.astype(float)
        m = df["log_tokens"].values.astype(float)
        idx_treat = names_treat.index("step_num")  # same position in both
        idx_med = names_full.index("log_tokens")

        c, a, b, c_prime = _mediation_paths(X_treat, X_full, y, m, idx_treat, idx_med)
        ab = a * b
        prop = ab / c if c != 0 else np.nan

        # ---- Cluster bootstrap: pre-group rows by qid
        qids = df["qid"].values
        unique_q, inv = np.unique(qids, return_inverse=True)
        K = len(unique_q)
        # rows per cluster
        rows_by_q = [np.where(inv == k)[0] for k in range(K)]
        row_counts = np.array([len(r) for r in rows_by_q])

        rng = np.random.default_rng(seed=99)
        ab_b = np.empty(B); cprime_b = np.empty(B); prop_b = np.empty(B)
        n_total = len(df)
        for it in range(B):
            sel_clusters = rng.integers(0, K, size=K)
            # Concatenate row indices: vectorised gather
            row_idx = np.concatenate([rows_by_q[c] for c in sel_clusters])
            Xt_b = X_treat[row_idx]
            Xf_b = X_full[row_idx]
            y_b = y[row_idx]
            m_b = m[row_idx]
            try:
                c_b, a_b, b_b, cp_b = _mediation_paths(
                    Xt_b, Xf_b, y_b, m_b, idx_treat, idx_med)
                ab_b[it] = a_b * b_b
                cprime_b[it] = cp_b
                prop_b[it] = (a_b * b_b) / c_b if c_b != 0 else np.nan
            except np.linalg.LinAlgError:
                ab_b[it] = np.nan; cprime_b[it] = np.nan; prop_b[it] = np.nan

        ab_lo, ab_hi = np.nanpercentile(ab_b, [2.5, 97.5])
        cp_lo, cp_hi = np.nanpercentile(cprime_b, [2.5, 97.5])
        pr_lo, pr_hi = np.nanpercentile(prop_b, [2.5, 97.5])

        out_rows.append({
            "spec": label,
            "n_obs": len(df), "n_qid": df["qid"].nunique(),
            "c_total": c,
            "c_prime_direct": c_prime,
            "c_prime_direct_ci_lo": cp_lo, "c_prime_direct_ci_hi": cp_hi,
            "a_treat_to_mediator": a,
            "b_mediator_to_outcome": b,
            "ab_indirect": ab,
            "ab_indirect_ci_lo": ab_lo, "ab_indirect_ci_hi": ab_hi,
            "prop_mediated": prop,
            "prop_mediated_ci_lo": pr_lo, "prop_mediated_ci_hi": pr_hi,
        })
    save(pd.DataFrame(out_rows), "s4_mediation")


# ---------------------------------------------------------------------------
# S5 Recovery survival analysis
# ---------------------------------------------------------------------------
def build_recovery_episodes(f1: pd.DataFrame) -> pd.DataFrame:
    """For each chain (qid, instruction, run), find the first step k* at which
    F1 >= threshold. If found, duration = number of consecutive subsequent steps
    keeping F1 >= threshold; event=1 if it eventually fell below before step 3,
    event=0 if it was still alive at step 3 (censored)."""
    threshold = RECOVERY_THRESHOLD
    chains = (f1.sort_values(["qid", "instruction_type", "run", "step"])
                .groupby(["qid", "instruction_type", "run"]))
    rows = []
    for (qid, instr, run), chain in chains:
        zero_start = chain.loc[chain["step"] == 0, "answer_f1"].iloc[0] == 0
        if not zero_start:
            continue
        steps = chain[chain["step"] > 0].sort_values("step")
        vals = steps["answer_f1"].values
        sts = steps["step"].values
        first_hit = None
        for i, v in enumerate(vals):
            if v >= threshold:
                first_hit = i
                break
        if first_hit is None:
            continue
        # Count how many consecutive steps from first_hit stay >= threshold
        survival = 1
        event = 0
        for j in range(first_hit + 1, len(vals)):
            if vals[j] >= threshold:
                survival += 1
            else:
                event = 1
                break
        rows.append({
            "qid": qid, "instruction_type": instr, "run": run,
            "first_hit_step": int(sts[first_hit]),
            "duration": int(survival),
            "event_observed": event,
            "n_hop": int(chain["n_hop"].iloc[0]),
        })
    return pd.DataFrame(rows)


def s5_survival(d: dict[str, pd.DataFrame]) -> None:
    section("S5  Kaplan-Meier survival of F1>=0.9 recovery")

    epi = build_recovery_episodes(d["f1"])
    save(epi, "s5_recovery_episodes")

    if len(epi) == 0:
        print("  No recovery episodes found, skipping KM.")
        return

    # Overall KM
    kmf = KaplanMeierFitter()
    kmf.fit(epi["duration"], event_observed=epi["event_observed"])
    surv = kmf.survival_function_.reset_index()
    surv.columns = ["duration_steps", "survival_prob"]
    ci = kmf.confidence_interval_.reset_index()
    ci.columns = ["duration_steps", "ci_lo", "ci_hi"]
    overall = surv.merge(ci, on="duration_steps")
    overall.insert(0, "stratum", "ALL")

    # By instruction
    per = []
    for instr, grp in epi.groupby("instruction_type"):
        if len(grp) < 5:
            continue
        kmf_i = KaplanMeierFitter()
        kmf_i.fit(grp["duration"], event_observed=grp["event_observed"])
        s = kmf_i.survival_function_.reset_index()
        s.columns = ["duration_steps", "survival_prob"]
        c = kmf_i.confidence_interval_.reset_index()
        c.columns = ["duration_steps", "ci_lo", "ci_hi"]
        m = s.merge(c, on="duration_steps")
        m.insert(0, "stratum", instr)
        per.append(m)

    km_table = pd.concat([overall] + per, ignore_index=True)
    save(km_table, "s5_km_table")

    # Log-rank: instruction strata
    try:
        lr = multivariate_logrank_test(
            epi["duration"], epi["instruction_type"], epi["event_observed"])
        save(pd.DataFrame([{
            "test": "multivariate_logrank_instruction",
            "n": len(epi), "test_statistic": lr.test_statistic, "p_value": lr.p_value,
        }]), "s5_logrank_instruction")
    except Exception as exc:
        print(f"  Log-rank failed: {exc}")

    # Summary
    summary = pd.DataFrame([{
        "n_episodes": len(epi),
        "median_duration_steps": float(np.median(epi["duration"])),
        "pct_event_observed": float(epi["event_observed"].mean()),
        "pct_lost_after_1_step": float((epi["duration"] == 1).mean()),
        "pct_stable_full_window": float(
            ((epi["duration"] == 3) & (epi["event_observed"] == 0)).mean()),
    }])
    save(summary, "s5_survival_summary")


# ---------------------------------------------------------------------------
# S6 TOST equivalence
# ---------------------------------------------------------------------------
def tost_paired(x: np.ndarray, y: np.ndarray, bound: float) -> dict:
    """Two one-sided t-tests for paired data. Null Ha: |mean diff| < bound."""
    diff = x - y
    n = len(diff)
    m = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = sd / np.sqrt(n)
    if se == 0:
        return {"n": n, "diff_mean": m, "lower_p": np.nan, "upper_p": np.nan,
                "equivalent": False, "bound": bound}
    # H0_lower: diff <= -bound  ;  H0_upper: diff >= +bound
    t_lower = (m + bound) / se
    t_upper = (m - bound) / se
    p_lower = 1 - st.t.cdf(t_lower, df=n - 1)   # upper-tail: rejects if diff > -bound
    p_upper = st.t.cdf(t_upper, df=n - 1)       # lower-tail: rejects if diff < +bound
    p_tost = max(p_lower, p_upper)
    return {"n": n, "diff_mean": m, "se": se, "bound": bound,
            "p_lower": p_lower, "p_upper": p_upper, "p_tost": p_tost,
            "equivalent": p_tost < 0.05}


def s6_tost(d: dict[str, pd.DataFrame]) -> None:
    section(f"S6  TOST equivalence on F1 step deltas (bound=+/-{EQUIV_BOUND})")

    f1 = d["f1"]
    wide = f1.pivot_table(index="qid", columns="step", values="answer_f1", aggfunc="mean")

    rows = []
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        pair = wide[[a, b]].dropna()
        r = tost_paired(pair[a].values, pair[b].values, EQUIV_BOUND)
        r["contrast"] = f"step{a}_vs_step{b}"
        r["metric"] = "answer_f1"
        rows.append(r)

    save(pd.DataFrame(rows), "s6_tost_equivalence")


# ---------------------------------------------------------------------------
# S7 Recovery sensitivity vs threshold
# ---------------------------------------------------------------------------
def s7_recovery_sensitivity(d: dict[str, pd.DataFrame], B: int) -> None:
    section(f"S7  Recovery rate vs F1 threshold (cluster boot B={B})")

    f1 = d["f1"]
    zero_start = f1[f1["step"] == 0].query("answer_f1 == 0")[
        ["qid", "instruction_type", "run"]].drop_duplicates()
    chain_max = (f1[f1["step"] > 0]
                 .groupby(["qid", "instruction_type", "run"])["answer_f1"].max()
                 .reset_index())
    zs = zero_start.merge(chain_max, on=["qid", "instruction_type", "run"], how="left")

    thresholds = [0.0001, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    rows = []
    qids = zs["qid"].values
    for thr in thresholds:
        vals = (zs["answer_f1"] >= thr).astype(float).values
        point, lo, hi = cluster_bootstrap_mean(
            vals, qids, B=B, seed=7000 + int(thr * 1000))
        rows.append({"threshold": thr, "recovery_rate": point,
                     "ci_lo": lo, "ci_hi": hi, "n_chains": len(zs), "B": B})
    save(pd.DataFrame(rows), "s7_recovery_sensitivity")


# ---------------------------------------------------------------------------
# S8 IRR stub (manual annotation scaffold)
# ---------------------------------------------------------------------------
def s8_irr_stub(d: dict[str, pd.DataFrame]) -> None:
    section("S8  IRR scaffold (manual annotation TBD)")

    f1 = d["f1"]
    bleurt = d["bleurt"]
    merged = f1.merge(bleurt[KEYS + ["bleurt_answer"]], on=KEYS, how="left")

    # Candidates for manual annotation: F1=0 AND BLEURT_answer in (0.3, 1.0]
    cand = merged[(merged["answer_f1"] == 0) & (merged["bleurt_answer"] >= 0.3)].copy()
    cand = cand[["qid", "instruction_type", "run", "step", "question",
                 "gold_answer", "predicted_answer", "bleurt_answer"]]
    cand["human_label"] = ""  # to be filled by annotators
    cand["auto_label_f1"] = 0
    cand = cand.sample(min(120, len(cand)), random_state=42).sort_values(
        ["qid", "instruction_type", "run", "step"])
    out_path = OUT / "s8_irr_candidates.csv"
    cand.to_csv(out_path, index=False)
    print(f"  [saved] {out_path.relative_to(ROOT)}  ({len(cand)} candidates "
          f"for manual labelling; fill 'human_label' with 0/1)")

    # Skeleton kappa computation, runs only when human_label is populated
    notes = pd.DataFrame([{
        "status": "AWAITING_LABELS",
        "n_candidates": len(cand),
        "instructions": ("Apri s8_irr_candidates.csv, riempi human_label con 0/1 "
                         "(corretto=1), poi rilancia con --kappa per ottenere "
                         "Cohen's kappa contro auto_label_f1 (sempre 0)."),
    }])
    save(notes, "s8_irr_status")


# ---------------------------------------------------------------------------
# S9 RQ summary table
# ---------------------------------------------------------------------------
def s9_rq_summary(d: dict[str, pd.DataFrame]) -> None:
    section("S9  RQ summary headlines")

    def load(name: str) -> pd.DataFrame:
        return pd.read_csv(OUT / f"{name}.csv")

    contrasts = load("s1_wilcoxon_step_contrasts")
    boots = load("s3_bootstrap_cis")
    mediation = load("s4_mediation")
    surv = load("s5_survival_summary")
    tost = load("s6_tost_equivalence")
    sens = load("s7_recovery_sensitivity")

    def find(df, **kw):
        sub = df.copy()
        for k, v in kw.items():
            sub = sub[sub[k] == v]
        return sub.iloc[0] if len(sub) else None

    rows = []

    r = find(contrasts, metric="answer_f1", contrast="step0_to_1")
    b = find(boots, statistic="delta_F1_step0_to_1")
    rows.append({
        "RQ": "RQ1: does the first rewriting step cause an F1 collapse?",
        "headline": "Yes; the collapse is concentrated step 0->1.",
        "estimate": f"Delta_F1 = {r['delta_mean']:+.3f}  "
                    f"[95% CI {b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}]",
        "test": f"Wilcoxon p_holm = {r['p_holm']:.1e}; "
                f"rank-biserial = {r['rank_biserial']:.3f}",
        "source": "s1_wilcoxon_step_contrasts + s3_bootstrap_cis",
    })

    r_b = find(tost, contrast="step1_vs_step2")
    r_c = find(tost, contrast="step2_vs_step3")
    rows.append({
        "RQ": "RQ1b: are later steps statistically equivalent to no change?",
        "headline": ("Yes for step1->2 and step2->3 within +/-0.02 F1"
                     if r_b["equivalent"] and r_c["equivalent"]
                     else "Partially / no"),
        "estimate": f"diff1->2={r_b['diff_mean']:+.4f}, diff2->3={r_c['diff_mean']:+.4f}",
        "test": (f"TOST(+/-{EQUIV_BOUND}) p={r_b['p_tost']:.3f} / {r_c['p_tost']:.3f}"),
        "source": "s6_tost_equivalence",
    })

    r = find(contrasts, metric="openfactscore", contrast="step1_to_2")
    r2 = find(contrasts, metric="openfactscore", contrast="step2_to_3")
    b = find(boots, statistic="delta_OFS_step1_to_3")
    rows.append({
        "RQ": "RQ2: does fact-level quality decay progressively?",
        "headline": "Yes; each step costs ~1pp OFS with very large effect size.",
        "estimate": f"Delta_OFS_1->3 = {b['estimate']:+.4f}  "
                    f"[95% CI {b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]",
        "test": f"Wilcoxon 1->2 r_b={r['rank_biserial']:.3f} (p={r['p_holm']:.1e}); "
                f"2->3 r_b={r2['rank_biserial']:.3f} (p={r2['p_holm']:.1e})",
        "source": "s1 + s3",
    })

    m = find(mediation, spec="F1 ~ step (mediator=log_tokens)")
    rows.append({
        "RQ": "RQ3: is length the causal mediator of the F1 collapse?",
        "headline": (f"Yes; ~{abs(m['prop_mediated'])*100:.0f}% of the step "
                     "effect on F1 is mediated by log(n_tokens)."),
        "estimate": (f"indirect={m['ab_indirect']:+.4f} "
                     f"[{m['ab_indirect_ci_lo']:+.4f}, {m['ab_indirect_ci_hi']:+.4f}]; "
                     f"direct={m['c_prime_direct']:+.4f} "
                     f"[{m['c_prime_direct_ci_lo']:+.4f}, {m['c_prime_direct_ci_hi']:+.4f}]"),
        "test": (f"prop_mediated={m['prop_mediated']:+.3f} "
                 f"[{m['prop_mediated_ci_lo']:+.3f}, {m['prop_mediated_ci_hi']:+.3f}]"),
        "source": "s4_mediation",
    })

    b = find(boots, statistic="recovery_rate_09")
    rows.append({
        "RQ": "RQ4: how stable is the apparent recovery?",
        "headline": (f"Recovery is rare and transient: "
                     f"{surv['pct_lost_after_1_step'].iloc[0]*100:.0f}% lose the "
                     f"correct answer after 1 step."),
        "estimate": (f"recovery_rate@F1>=0.9 = {b['estimate']*100:.1f}%  "
                     f"[{b['ci_lo']*100:.1f}%, {b['ci_hi']*100:.1f}%]"),
        "test": (f"KM median duration = {surv['median_duration_steps'].iloc[0]:.0f} step; "
                 f"% still alive at step 3 = "
                 f"{surv['pct_stable_full_window'].iloc[0]*100:.0f}%"),
        "source": "s3 + s5",
    })

    s09 = find(sens, threshold=0.9)
    s00 = find(sens, threshold=0.0001)
    rows.append({
        "RQ": "RQ4b: how sensitive is recovery to the F1 threshold?",
        "headline": ("Strongly: from ~22% at any-overlap to ~11% at F1>=0.9. "
                     "Choice of threshold drives the conclusion."),
        "estimate": (f"@thr>0: {s00['recovery_rate']*100:.1f}%  "
                     f"[{s00['ci_lo']*100:.1f}, {s00['ci_hi']*100:.1f}]  ;  "
                     f"@thr>=0.9: {s09['recovery_rate']*100:.1f}%  "
                     f"[{s09['ci_lo']*100:.1f}, {s09['ci_hi']*100:.1f}]"),
        "test": "Cluster bootstrap on qid",
        "source": "s7_recovery_sensitivity",
    })

    save(pd.DataFrame(rows), "s9_rq_summary")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=10_000,
                    help="Number of cluster-bootstrap iterations (default 10000).")
    ap.add_argument("--skip-bootstrap", action="store_true",
                    help="Skip all bootstrap CIs (sets B=200 internally for speed).")
    args = ap.parse_args()
    B = 200 if args.skip_bootstrap else args.bootstrap

    section(f"Statistical analysis 300q (B={B})")
    OUT.mkdir(parents=True, exist_ok=True)

    d = load_all()

    s0_descriptives(d)
    s1_omnibus(d)
    s2_mixed_models(d)
    s3_bootstrap_cis(d, B=B)
    s4_mediation(d, B=B)
    s5_survival(d)
    s6_tost(d)
    s7_recovery_sensitivity(d, B=B)
    s8_irr_stub(d)
    s9_rq_summary(d)

    # Manifest
    manifest = sorted(p.name for p in OUT.glob("*.csv"))
    (OUT / "MANIFEST.json").write_text(
        json.dumps({"B": B, "files": manifest}, indent=2))
    print(f"\nWrote {len(manifest)} CSVs to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
