"""
Inference tests sul run 300q.

Pipeline di analisi statistica gerarchica per le metriche del rewriting iterativo:
- Answer F1 (OLMo-3.1 4-bit)         step 0-3, distribuzione bimodale -> GLMM logistico
- OpenFActScore (4-bit)              step 1-3, LMM
- BERTScore baseline (vs step 0)     step 1-3 (lo step 0 e' implicito nel confronto), LMM
- BERTScore consecutive (vs k-1)     step 1-3, LMM

Struttura random: random intercept per qid (+ eventualmente run come (1|qid:run)).
Instruction nested in group -> modelli step x instruction_type fittati separatamente
per content e style.

Output: results/300q/stats/inference/*.csv + inference_summary.{csv,md}.

Uso:
    python3.11 scripts/300q/inference_tests.py
    python3.11 scripts/300q/inference_tests.py --skip-bootstrap
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "300q"
OUT_DIR = RESULTS_DIR / "stats" / "inference"
DIAG_DIR = RESULTS_DIR / "stats" / "diagnostics"

FILES = {
    "chains": RESULTS_DIR / "rewriting_chains_300q.csv",
    "f1": RESULTS_DIR / "rewriting_chains_300q_answer_f1_olmo31_4bit.csv",
    "ofs": RESULTS_DIR / "rewriting_chains_300q_openfactscore.csv",
    "bs": RESULTS_DIR / "rewriting_chains_300q_bertscore.csv",
}

KEYS = ["qid", "group", "instruction_type", "run", "step"]


# ---------- I/O helpers ----------

def _section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def _save(df: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  [saved] {path.relative_to(ROOT)}")


def _add_hop(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_hop"] = df["qid"].str.extract(r"^(\d+)hop")[0].astype("Int64")
    return df


def load_data() -> dict[str, pd.DataFrame]:
    """Task 1 - load and filter step coverage per metric."""
    chains = _add_hop(pd.read_csv(FILES["chains"]))
    f1 = _add_hop(pd.read_csv(FILES["f1"]))
    ofs = _add_hop(pd.read_csv(FILES["ofs"]))
    bs = _add_hop(pd.read_csv(FILES["bs"]))

    f1["f1_bin"] = (f1["answer_f1"] > 0).astype(int)

    # propaga n_tokens (e orig_tokens) ai dataframe delle metriche
    orig_tokens = (chains[chains["step"] == 0]
                   .groupby("qid")["n_tokens"].mean().rename("orig_tokens"))
    chains_tok = chains[KEYS + ["n_tokens"]].merge(orig_tokens, on="qid", how="left")
    f1 = f1.merge(chains_tok, on=KEYS, how="left")
    ofs = ofs.merge(chains_tok, on=KEYS, how="left")
    bs = bs.merge(chains_tok, on=KEYS, how="left")

    keep_cols = KEYS + ["n_hop", "n_tokens", "orig_tokens"]
    bs_base = bs[keep_cols + ["bert_f1_baseline"]].copy()
    bs_cons = bs.loc[bs["step"] >= 1, keep_cols + ["bert_f1_consecutive"]].copy()

    print(f"F1            n={len(f1):>6}  steps={sorted(f1['step'].unique())}  hops={sorted(f1['n_hop'].dropna().unique())}")
    print(f"OFS           n={len(ofs):>6}  steps={sorted(ofs['step'].unique())}  hops={sorted(ofs['n_hop'].dropna().unique())}")
    print(f"BERT base     n={len(bs_base):>6}  steps={sorted(bs_base['step'].unique())}  hops={sorted(bs_base['n_hop'].dropna().unique())}")
    print(f"BERT cons     n={len(bs_cons):>6}  steps={sorted(bs_cons['step'].unique())}  hops={sorted(bs_cons['n_hop'].dropna().unique())}")

    return {"f1": f1, "ofs": ofs, "bs_base": bs_base, "bs_cons": bs_cons, "chains": chains}


# ---------- Task 2 - diagnostics ----------

def diagnostics(data: dict[str, pd.DataFrame]) -> None:
    """Shapiro on residuals of a null model + ICC per qid."""
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    cfg = [
        ("f1", "answer_f1", "f1"),
        ("ofs", "factscore", "ofs"),
        ("bs_base", "bert_f1_baseline", "bs_base"),
        ("bs_cons", "bert_f1_consecutive", "bs_cons"),
    ]
    for key, col, _ in cfg:
        df = data[key].dropna(subset=[col]).copy()
        df["qid_c"] = df["qid"].astype("category")
        null = smf.mixedlm(f"{col} ~ 1", df, groups=df["qid_c"]).fit(method="lbfgs")
        resid = null.resid
        sample = resid.sample(min(5000, len(resid)), random_state=0)
        W, p = st.shapiro(sample)
        var_re = float(null.cov_re.iloc[0, 0])
        var_resid = float(null.scale)
        icc = var_re / (var_re + var_resid) if (var_re + var_resid) > 0 else float("nan")
        rows.append({
            "metric": key, "n": len(df),
            "mean": df[col].mean(), "std": df[col].std(),
            "shapiro_W": W, "shapiro_p": p,
            "var_qid": var_re, "var_resid": var_resid, "icc_qid": icc,
        })
        print(f"  {key:<8} ICC(qid)={icc:.3f}  Shapiro W={W:.3f} p={p:.2e}")
    diag = pd.DataFrame(rows)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    diag.to_csv(DIAG_DIR / "diagnostics.csv", index=False)
    print(f"  [saved] {(DIAG_DIR / 'diagnostics.csv').relative_to(ROOT)}")


# ---------- model fitting helpers ----------

def _fit_lmm(df: pd.DataFrame, formula: str, group_col: str = "qid"):
    df = df.copy()
    df["_grp"] = df[group_col].astype("category")
    return smf.mixedlm(formula, df, groups=df["_grp"]).fit(method="lbfgs")


def _fit_glmm_logit(df: pd.DataFrame, formula: str, group_col: str = "qid"):
    df = df.copy()
    df["_grp"] = df[group_col].astype("category")
    return smf.mixedlm(formula, df, groups=df["_grp"]).fit(method="lbfgs")


def _pairwise_steps(df: pd.DataFrame, metric: str, kind: str, steps: list[int]) -> pd.DataFrame:
    """Planned contrasts step k vs step k+1 within qid; Holm-adjusted."""
    rows = []
    for a, b in zip(steps[:-1], steps[1:]):
        sub = df[df["step"].isin([a, b])]
        wide = sub.groupby(["qid", "step"])[metric].mean().unstack("step").dropna()
        if kind == "logit":
            wide_a, wide_b = (wide[a] > 0).astype(int), (wide[b] > 0).astype(int)
            stat, p = st.wilcoxon(wide_a, wide_b, zero_method="zsplit")
        else:
            stat, p = st.wilcoxon(wide[a], wide[b], zero_method="zsplit")
        diff = (wide[b] - wide[a]).mean()
        rb = 1 - 2 * stat / (len(wide) * (len(wide) + 1) / 2)
        rows.append({"contrast": f"step{a}->step{b}", "n_pairs": len(wide),
                     "mean_diff": diff, "wilcoxon_stat": stat, "p": p, "rank_biserial": rb})
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p"], method="holm")[1]
    return out


def _format_coefs(result, label: str) -> pd.DataFrame:
    params = result.params
    bse = result.bse
    tvals = result.tvalues
    pvals = result.pvalues
    df = pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "std_err": bse.reindex(params.index).values,
        "z": tvals.reindex(params.index).values,
        "p": pvals.reindex(params.index).values,
    })
    df.insert(0, "model", label)
    return df


# ---------- Task 3 - Answer F1 vs step ----------

def test_f1_step(data: dict) -> dict:
    _section("Test 1 - Answer F1 vs step (GLMM logit on F1>0)")
    df = data["f1"]
    m = _fit_glmm_logit(df, "f1_bin ~ C(step)")
    print(m.summary().tables[1])
    coefs = _format_coefs(m, "F1_step_GLMM_logit")
    contrasts = _pairwise_steps(df, "f1_bin", "logit", [0, 1, 2, 3])
    _save(coefs, "f1_step_model")
    _save(contrasts, "f1_step_contrasts")
    return {"model": m, "contrasts": contrasts}


# ---------- Task 4 - OFS vs step ----------

def test_ofs_step(data: dict) -> dict:
    _section("Test 2 - OpenFActScore vs step (LMM, step 1-3)")
    df = data["ofs"]
    m = _fit_lmm(df, "factscore ~ C(step)")
    print(m.summary().tables[1])
    coefs = _format_coefs(m, "OFS_step_LMM")
    contrasts = _pairwise_steps(df, "factscore", "lmm", [1, 2, 3])
    _save(coefs, "ofs_step_model")
    _save(contrasts, "ofs_step_contrasts")
    return {"model": m, "contrasts": contrasts}


# ---------- Task 5 - BERTScore vs step ----------

def test_bs_step(data: dict) -> dict:
    _section("Test 3 - BERTScore vs step")
    out = {}
    print("\n[baseline vs step 0]")
    m_b = _fit_lmm(data["bs_base"], "bert_f1_baseline ~ C(step)")
    print(m_b.summary().tables[1])
    _save(_format_coefs(m_b, "BS_baseline_step_LMM"), "bs_baseline_step_model")
    _save(_pairwise_steps(data["bs_base"], "bert_f1_baseline", "lmm", [1, 2, 3]),
          "bs_baseline_step_contrasts")

    print("\n[consecutive vs step k-1]")
    m_c = _fit_lmm(data["bs_cons"], "bert_f1_consecutive ~ C(step)")
    print(m_c.summary().tables[1])
    _save(_format_coefs(m_c, "BS_consecutive_step_LMM"), "bs_consecutive_step_model")
    _save(_pairwise_steps(data["bs_cons"], "bert_f1_consecutive", "lmm", [1, 2, 3]),
          "bs_consecutive_step_contrasts")

    out["baseline"], out["consecutive"] = m_b, m_c
    return out


# ---------- Task 6 - step x instruction_type (nested in group) ----------

def test_step_x_instruction(data: dict) -> dict:
    _section("Test 4 - step x instruction_type (separato per content/style)")
    out = {}
    cfg = [
        ("f1", "f1_bin", "F1_logit"),
        ("ofs", "factscore", "OFS_LMM"),
        ("bs_base", "bert_f1_baseline", "BS_base_LMM"),
        ("bs_cons", "bert_f1_consecutive", "BS_cons_LMM"),
    ]
    for key, col, label in cfg:
        df = data[key]
        for g in ["content", "style"]:
            sub = df[df["group"] == g]
            if sub.empty:
                continue
            m = _fit_lmm(sub, f"{col} ~ C(step) * C(instruction_type)")
            tag = f"{label}_{g}_step_x_instr"
            print(f"\n[{tag}]")
            print(m.summary().tables[1])
            _save(_format_coefs(m, tag), tag.lower())
            out[tag] = m
    return out


# ---------- Task 7 - step x group ----------

def test_step_x_group(data: dict) -> dict:
    _section("Test 5 - step x group")
    out = {}
    cfg = [
        ("f1", "f1_bin", "F1_logit", [1, 2, 3]),
        ("ofs", "factscore", "OFS_LMM", [1, 2, 3]),
        ("bs_base", "bert_f1_baseline", "BS_base_LMM", [1, 2, 3]),
        ("bs_cons", "bert_f1_consecutive", "BS_cons_LMM", [1, 2, 3]),
    ]
    for key, col, label, steps in cfg:
        df = data[key]
        sub = df[df["step"].isin(steps)]
        m = _fit_lmm(sub, f"{col} ~ C(step) * C(group)")
        tag = f"{label}_step_x_group"
        print(f"\n[{tag}]  (steps={steps})")
        print(m.summary().tables[1])
        _save(_format_coefs(m, tag), tag.lower())
        out[tag] = m
    return out


# ---------- Task 8 - run variability ----------

def test_run_variance(data: dict) -> dict:
    """Variance decomposition empirica: var qid / var(run within qid) / var residuo.

    Per ogni metrica residualizziamo l'effetto fisso di step (riduzione del modello),
    poi decomponiamo:
      var_qid     = Var(media-per-qid)
      var_run     = Var(media-per-(qid,run)) intra-qid
      var_resid   = varianza residua intra-(qid,run)
    e calcoliamo gli ICC. LRT su (1|qid) vs (1|qid:run) come test di significativita
    della componente run.
    """
    _section("Test 6 - variabilita tra run")
    rows = []
    cfg = [("f1", "answer_f1"), ("ofs", "factscore"),
           ("bs_base", "bert_f1_baseline"), ("bs_cons", "bert_f1_consecutive")]
    for key, col in cfg:
        df = data[key].dropna(subset=[col]).copy()
        # rimuovi effetto fisso di step
        df["resid"] = df[col] - df.groupby("step")[col].transform("mean")

        qid_means = df.groupby("qid")["resid"].mean()
        qid_run_means = df.groupby(["qid", "run"])["resid"].mean()
        # var run-within-qid: per ogni qid, varianza tra medie di run, poi media sui qid
        run_within_qid = (qid_run_means - qid_run_means.groupby(level=0).transform("mean"))
        within_run = df["resid"] - df.set_index(["qid", "run"]).index.map(qid_run_means)

        var_qid = float(qid_means.var(ddof=1))
        var_run = float(run_within_qid.var(ddof=1))
        var_res = float(np.var(within_run, ddof=1))
        total = var_qid + var_run + var_res
        icc_qid = var_qid / total if total > 0 else float("nan")
        icc_run = var_run / total if total > 0 else float("nan")

        # Friedman su run-within-qid medi (per ogni qid, 3 medie di run -> test omnibus)
        wide = (df.groupby(["qid", "run"])[col].mean().unstack("run").dropna())
        if wide.shape[1] >= 2:
            stat, p = st.friedmanchisquare(*[wide[c] for c in wide.columns])
        else:
            stat, p = float("nan"), float("nan")

        rows.append({"metric": key, "friedman_chi2": stat, "friedman_p": p,
                     "var_qid": var_qid, "var_run": var_run, "var_resid": var_res,
                     "icc_qid": icc_qid, "icc_run": icc_run})
        print(f"  {key:<8} Friedman(run) chi2={stat:6.2f} p={p:.3g}  "
              f"ICC qid={icc_qid:.3f}  ICC run={icc_run:.4f}")
    out = pd.DataFrame(rows)
    _save(out, "run_variance_decomposition")
    return {"table": out}


# ---------- Task 9 - correlation between metrics ----------

def test_correlation_metrics(data: dict) -> dict:
    _section("Test 7 - correlazione tra metriche (mixed-effects)")
    # join sui campi chiave; OFS solo step 1-3 -> intersezione naturale
    keys = ["qid", "group", "instruction_type", "run", "step"]
    df = data["f1"][keys + ["answer_f1", "f1_bin"]].merge(
        data["ofs"][keys + ["factscore"]], on=keys, how="inner"
    ).merge(
        data["bs_base"][keys + ["bert_f1_baseline"]], on=keys, how="inner"
    ).merge(
        data["bs_cons"][keys + ["bert_f1_consecutive"]], on=keys, how="inner"
    )
    print(f"  merged rows: {len(df)}")

    m1 = _fit_lmm(df, "answer_f1 ~ factscore + bert_f1_baseline + C(step)")
    print("\n[F1 ~ OFS + BERTbase + step | (1|qid)]")
    print(m1.summary().tables[1])
    _save(_format_coefs(m1, "F1_on_OFS_BS_LMM"), "corr_f1_on_ofs_bs")

    # repeated-measures correlations
    rm_pairs = [
        ("answer_f1", "factscore"),
        ("answer_f1", "bert_f1_baseline"),
        ("factscore", "bert_f1_baseline"),
    ]
    rm_rows = []
    for a, b in rm_pairs:
        r = pg.rm_corr(df, x=a, y=b, subject="qid")
        ci = r["CI95"].iloc[0] if "CI95" in r.columns else r["CI95%"].iloc[0]
        rm_rows.append({"x": a, "y": b, "r": float(r["r"].iloc[0]),
                        "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                        "p": float(r["pval"].iloc[0])})
    rm_out = pd.DataFrame(rm_rows)
    print("\n[repeated-measures correlations]")
    print(rm_out.round(3).to_string(index=False))
    _save(rm_out, "corr_repeated_measures")
    return {"lmm": m1, "rm_corr": rm_out}


# ---------- Task 10 - robustness checks ----------

def test_robustness(data: dict) -> dict:
    _section("Test 8 - robustness non-parametrici (Friedman su step)")
    rows = []
    cfg = [("f1", "answer_f1", [0, 1, 2, 3]),
           ("ofs", "factscore", [1, 2, 3]),
           ("bs_base", "bert_f1_baseline", [1, 2, 3]),
           ("bs_cons", "bert_f1_consecutive", [1, 2, 3])]
    for key, col, steps in cfg:
        df = data[key]
        # collapse over run/instruction within qid x step (mean) to get within-subject array
        wide = (df.groupby(["qid", "step"])[col].mean()
                .unstack("step").dropna())
        wide = wide[steps]
        stat, p = st.friedmanchisquare(*[wide[s] for s in steps])
        rows.append({"metric": key, "steps": str(steps),
                     "friedman_chi2": stat, "df": len(steps) - 1, "p": p,
                     "n_qid": len(wide)})
        print(f"  {key:<8} Friedman chi2={stat:.2f} df={len(steps)-1} p={p:.3g} (n={len(wide)})")
    out = pd.DataFrame(rows)
    _save(out, "robustness_friedman")
    return {"friedman": out}


# ---------- Test 9 - RQ2b: effetto della complessita (n_hop) ----------

def test_hop_effect(data: dict) -> dict:
    """RQ2b: la complessita della domanda (2/3/4-hop) modula il degrado?

    Modelli: metric ~ C(step) * C(n_hop) + (1|qid). Riporta anche medie per
    hop x step. Per OFS, se presente solo un hop, salta con warning.
    """
    _section("Test 9 (RQ2b) - effetto della complessita (n_hop)")
    out = {}
    cfg = [
        ("f1", "f1_bin", "F1_logit"),
        ("ofs", "factscore", "OFS_LMM"),
        ("bs_base", "bert_f1_baseline", "BS_base_LMM"),
        ("bs_cons", "bert_f1_consecutive", "BS_cons_LMM"),
    ]
    summary_rows = []
    for key, col, label in cfg:
        df = data[key].dropna(subset=[col, "n_hop"]).copy()
        df["n_hop"] = df["n_hop"].astype(int)
        hops = sorted(df["n_hop"].unique().tolist())
        if len(hops) < 2:
            print(f"  [skip] {key}: un solo livello di n_hop ({hops}) - non testabile")
            continue
        print(f"\n[{label} - n_hop levels: {hops}]")

        # medie per hop x step
        means = df.groupby(["n_hop", "step"])[col].mean().unstack("step").round(3)
        print(means)

        # modello con interazione
        m = _fit_lmm(df, f"{col} ~ C(step) * C(n_hop)")
        print(m.summary().tables[1])
        coefs = _format_coefs(m, f"{label}_step_x_hop")
        _save(coefs, f"{label.lower()}_step_x_hop")

        # rate di degrado per hop: contrasto step_max vs step_min per ciascun hop
        step_min, step_max = min(df["step"].unique()), max(df["step"].unique())
        per_hop = []
        for h in hops:
            sub = df[df["n_hop"] == h]
            wide = sub.groupby(["qid", "step"])[col].mean().unstack("step").dropna()
            if step_min in wide.columns and step_max in wide.columns:
                diff = (wide[step_max] - wide[step_min])
                try:
                    stat, p = st.wilcoxon(wide[step_min], wide[step_max], zero_method="zsplit")
                except ValueError:
                    stat, p = float("nan"), float("nan")
                per_hop.append({"metric": key, "n_hop": int(h), "n_qid": len(wide),
                                "mean_at_min": wide[step_min].mean(),
                                "mean_at_max": wide[step_max].mean(),
                                "mean_diff_min_to_max": diff.mean(),
                                "wilcoxon_p": p})
        per_hop_df = pd.DataFrame(per_hop)
        if not per_hop_df.empty:
            per_hop_df["wilcoxon_p_holm"] = multipletests(per_hop_df["wilcoxon_p"].fillna(1), method="holm")[1]
            print("\n[degrado " + f"step{step_min}->step{step_max}" + " per hop]")
            print(per_hop_df.round(4).to_string(index=False))
            _save(per_hop_df, f"{label.lower()}_degradation_by_hop")
            summary_rows.append(per_hop_df)

        out[label] = m
    if summary_rows:
        out["degradation_by_hop"] = pd.concat(summary_rows, ignore_index=True)
    return out


# ---------- Test 10 - effetto della lunghezza (n_tokens) ----------

def test_length_effect(data: dict) -> dict:
    """Quanto la lunghezza del rewriting modula F1 e BERTScore?

    Tre blocchi:
      A) Descrittiva: n_tokens per step x instruction, % corte (<200) e saturate (>=2048)
      B) Predittore: F1 / BERTbase ~ log(n_tokens) + step + (1|qid)
      C) Mediation: aggiungere log(n_tokens) attenua l'effetto di instruction_type?
    """
    _section("Test 10 - effetto della lunghezza (n_tokens)")
    chains = data["chains"]
    out = {}

    # A) descrittiva
    desc = (chains[chains["step"] >= 1]
            .groupby(["instruction_type", "step"])["n_tokens"]
            .agg(["mean", "median", "std", "min", "max"]).round(1).reset_index())
    print("\n[A.1] n_tokens per instruction x step")
    print(desc.to_string(index=False))
    _save(desc, "length_descriptive")

    flags = chains[chains["step"] >= 1].copy()
    flags["short"] = (flags["n_tokens"] < 200).astype(int)
    flags["saturated"] = (flags["n_tokens"] >= 2048).astype(int)
    comp = (flags.groupby(["instruction_type", "step"])
            .agg(n=("short", "size"),
                 pct_short=("short", "mean"),
                 pct_saturated=("saturated", "mean"),
                 median_tokens=("n_tokens", "median"))
            .round(3).reset_index())
    print("\n[A.2] % catene corte (<200 tok) e saturate (>=2048 tok)")
    print(comp.to_string(index=False))
    _save(comp, "length_compliance")

    # B) predittore: F1 e BERTbase ~ log(n_tokens) + step + (1|qid)
    f1 = data["f1"].dropna(subset=["n_tokens", "answer_f1"]).copy()
    f1["log_ntokens"] = np.log(f1["n_tokens"].clip(lower=1))
    print("\n[B.1] F1 ~ log(n_tokens) + step + (1|qid) (tutte le istruzioni)")
    m_f1 = _fit_lmm(f1, "answer_f1 ~ log_ntokens + C(step)")
    print(m_f1.summary().tables[1])
    _save(_format_coefs(m_f1, "F1_on_logtokens"), "length_f1_on_logtokens")

    bs = data["bs_base"].dropna(subset=["n_tokens", "bert_f1_baseline"]).copy()
    bs["log_ntokens"] = np.log(bs["n_tokens"].clip(lower=1))
    print("\n[B.2] BERTbase ~ log(n_tokens) + step + (1|qid)")
    m_bs = _fit_lmm(bs, "bert_f1_baseline ~ log_ntokens + C(step)")
    print(m_bs.summary().tables[1])
    _save(_format_coefs(m_bs, "BSbase_on_logtokens"), "length_bs_on_logtokens")

    # C) mediation su F1 e BERTbase
    print("\n[C.1] F1 ~ C(instruction_type) + C(step) + (1|qid)  - SENZA n_tokens")
    f1_sub = f1[f1["step"] >= 1].copy()
    m_a = _fit_lmm(f1_sub, "answer_f1 ~ C(instruction_type) + C(step)")
    print(m_a.summary().tables[1])
    _save(_format_coefs(m_a, "F1_on_instr_no_tokens"), "length_f1_mediation_no_tokens")

    print("\n[C.2] F1 ~ C(instruction_type) + C(step) + log(n_tokens) + (1|qid)  - CON n_tokens")
    m_b = _fit_lmm(f1_sub, "answer_f1 ~ C(instruction_type) + C(step) + log_ntokens")
    print(m_b.summary().tables[1])
    _save(_format_coefs(m_b, "F1_on_instr_with_tokens"), "length_f1_mediation_with_tokens")

    print("\n[C.3] BERTbase ~ C(instruction_type) + C(step) + (1|qid)  - SENZA n_tokens")
    m_c = _fit_lmm(bs, "bert_f1_baseline ~ C(instruction_type) + C(step)")
    print(m_c.summary().tables[1])
    _save(_format_coefs(m_c, "BSbase_on_instr_no_tokens"), "length_bs_mediation_no_tokens")

    print("\n[C.4] BERTbase ~ C(instruction_type) + C(step) + log(n_tokens) + (1|qid)  - CON n_tokens")
    m_d = _fit_lmm(bs, "bert_f1_baseline ~ C(instruction_type) + C(step) + log_ntokens")
    print(m_d.summary().tables[1])
    _save(_format_coefs(m_d, "BSbase_on_instr_with_tokens"), "length_bs_mediation_with_tokens")

    out.update({"f1_logtok": m_f1, "bs_logtok": m_bs,
                "f1_med_no": m_a, "f1_med_yes": m_b,
                "bs_med_no": m_c, "bs_med_yes": m_d})
    return out


# ---------- Task 11 - summary ----------

def build_summary(results: dict) -> None:
    _section("Summary")
    rows = []

    for tag in ["f1_step_contrasts", "ofs_step_contrasts",
                "bs_baseline_step_contrasts", "bs_consecutive_step_contrasts"]:
        path = OUT_DIR / f"{tag}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                "family": tag.replace("_step_contrasts", ""),
                "test": "Wilcoxon paired (planned step contrast)",
                "contrast": r["contrast"],
                "n_pairs": r["n_pairs"],
                "mean_diff": round(r["mean_diff"], 4),
                "effect_size": round(r["rank_biserial"], 3),
                "p_raw": r["p"], "p_holm": r["p_holm"],
            })

    if (OUT_DIR / "run_variance_decomposition.csv").exists():
        df = pd.read_csv(OUT_DIR / "run_variance_decomposition.csv")
        for _, r in df.iterrows():
            rows.append({"family": r["metric"], "test": "Friedman over runs",
                         "contrast": "run (within qid)", "n_pairs": None,
                         "mean_diff": None, "effect_size": round(r["icc_run"], 4),
                         "p_raw": r["friedman_p"], "p_holm": None})

    if (OUT_DIR / "corr_repeated_measures.csv").exists():
        df = pd.read_csv(OUT_DIR / "corr_repeated_measures.csv")
        for _, r in df.iterrows():
            rows.append({"family": "metric-correlation", "test": "rm_corr",
                         "contrast": f"{r['x']} ~ {r['y']}", "n_pairs": None,
                         "mean_diff": None, "effect_size": round(r["r"], 3),
                         "p_raw": r["p"], "p_holm": None})

    if (OUT_DIR / "robustness_friedman.csv").exists():
        df = pd.read_csv(OUT_DIR / "robustness_friedman.csv")
        for _, r in df.iterrows():
            rows.append({"family": r["metric"], "test": "Friedman (step omnibus)",
                         "contrast": f"step in {r['steps']}", "n_pairs": r["n_qid"],
                         "mean_diff": None, "effect_size": None,
                         "p_raw": r["p"], "p_holm": None})

    for f in OUT_DIR.glob("*_degradation_by_hop.csv"):
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            rows.append({"family": f"hop-{r['metric']}",
                         "test": "Wilcoxon step_min vs step_max (per hop)",
                         "contrast": f"{r['n_hop']}-hop", "n_pairs": r["n_qid"],
                         "mean_diff": round(r["mean_diff_min_to_max"], 4),
                         "effect_size": None,
                         "p_raw": r["wilcoxon_p"], "p_holm": r["wilcoxon_p_holm"]})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "inference_summary.csv", index=False)

    md_lines = ["# 300q - inference summary\n",
                "Random structure: `(1|qid)`. Step contrasts: planned, Holm-adjusted.\n"]
    for family, sub in summary.groupby("family"):
        md_lines.append(f"\n## {family}\n")
        md_lines.append(sub.to_markdown(index=False))
        md_lines.append("\n")
    (OUT_DIR / "inference_summary.md").write_text("\n".join(md_lines))
    print(f"\n  [saved] {(OUT_DIR / 'inference_summary.csv').relative_to(ROOT)}")
    print(f"  [saved] {(OUT_DIR / 'inference_summary.md').relative_to(ROOT)}")


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-bootstrap", action="store_true")
    p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _section("Task 1 - load data")
    data = load_data()

    _section("Task 2 - diagnostics")
    diagnostics(data)

    results = {}
    results["f1_step"] = test_f1_step(data)
    results["ofs_step"] = test_ofs_step(data)
    results["bs_step"] = test_bs_step(data)
    results["step_x_instr"] = test_step_x_instruction(data)
    results["step_x_group"] = test_step_x_group(data)
    results["run_var"] = test_run_variance(data)
    results["corr"] = test_correlation_metrics(data)
    results["robust"] = test_robustness(data)
    results["hop"] = test_hop_effect(data)
    results["length"] = test_length_effect(data)
    build_summary(results)


if __name__ == "__main__":
    main()
