"""
Recovery analysis sul run 300q.

Indaga il fenomeno: chains che partono con F1=0 a step 0 (testo originale)
e poi recuperano (F1 > 0 in almeno uno step successivo).

Cinque test:
  1. GLMM logit: recovered ~ group + (1|qid)            -> group effect
  2. GLMM logit: recovered ~ instruction_type + (1|qid) -> instruction effect + pairwise
  3. GLMM logit: recovered ~ n_hop + (1|qid)            -> hop effect
     + Cochran-Armitage trend test
  4. GLMM logit con interazione + LRT                   -> hop x instruction
  5. Persistenza & step di recupero                     -> chi-quadro

Output: results/300q/stats/recovery/*.csv

Uso:
    python3.11 scripts/300q/recovery_analysis.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
F1_CSV = ROOT / "results" / "300q" / "rewriting_chains_300q_answer_f1_olmo31_4bit.csv"
OUT_DIR = ROOT / "results" / "300q" / "stats" / "recovery"


# ---------- helpers ----------

def _section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def _save(df: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}.csv"
    df.to_csv(p, index=False)
    print(f"  [saved] {p.relative_to(ROOT)}")


def _fit_glmm_logit(df: pd.DataFrame, formula: str, group: str = "qid"):
    """LMM su 0/1 (uso come approssimazione GLMM logit, vista la struttura)."""
    df = df.copy()
    df["_grp"] = df[group].astype("category")
    return smf.mixedlm(formula, df, groups=df["_grp"]).fit(method="lbfgs")


def _coefs(result, label: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "term": result.params.index,
        "coef": result.params.values,
        "std_err": result.bse.reindex(result.params.index).values,
        "z": result.tvalues.reindex(result.params.index).values,
        "p": result.pvalues.reindex(result.params.index).values,
    })
    out.insert(0, "model", label)
    return out


# ---------- preparazione dati ----------

def prepare(f1_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(f1_csv)
    df["n_hop"] = df["qid"].str.extract(r"^(\d+)hop")[0].astype(int)

    pivot = df.pivot_table(
        index=["qid", "group", "instruction_type", "run", "n_hop"],
        columns="step", values="answer_f1"
    ).reset_index()

    # filtra chains con F1=0 a step 0
    zero_start = pivot[pivot[0] == 0].copy()
    zero_start["max_later"] = zero_start[[1, 2, 3]].max(axis=1)
    zero_start["recovered"] = (zero_start["max_later"] > 0).astype(int)
    zero_start["n_pos_steps"] = (zero_start[[1, 2, 3]] > 0).sum(axis=1)

    def first_step(row):
        for s in (1, 2, 3):
            if row[s] > 0:
                return s
        return 0
    zero_start["first_recovery_step"] = zero_start.apply(first_step, axis=1)

    return zero_start


# ---------- Test 1 - group ----------

def test_group(zs: pd.DataFrame) -> None:
    _section("Test 1 - effetto del GROUP (content vs style)")
    print("\n[descrittiva]")
    desc = (zs.groupby("group")
            .agg(n_zero_start=("recovered", "size"),
                 n_recovered=("recovered", "sum"),
                 pct_recovered=("recovered", "mean")).round(3).reset_index())
    print(desc.to_string(index=False))
    _save(desc, "01_group_descriptive")

    m = _fit_glmm_logit(zs, "recovered ~ C(group)")
    print("\n[GLMM logit: recovered ~ C(group) + (1|qid)]")
    print(m.summary().tables[1])
    coefs = _coefs(m, "recovered_on_group")
    # OR per il coefficient di group
    coefs["odds_ratio"] = np.exp(coefs["coef"]) if "coef" in coefs.columns else None
    _save(coefs, "01_group_glmm")

    # chi2 di sanity
    tab = pd.crosstab(zs["group"], zs["recovered"])
    chi2, p, dof, _ = st.chi2_contingency(tab)
    print(f"\n[sanity chi2 2x2] chi2={chi2:.3f} dof={dof} p={p:.4f}")
    _save(pd.DataFrame([{"test": "chi2_2x2_group", "chi2": chi2, "dof": dof, "p": p}]),
          "01_group_chi2")


# ---------- Test 2 - instruction ----------

def test_instruction(zs: pd.DataFrame) -> None:
    _section("Test 2 - effetto dell'ISTRUZIONE")
    print("\n[descrittiva]")
    desc = (zs.groupby("instruction_type")
            .agg(n_zero_start=("recovered", "size"),
                 n_recovered=("recovered", "sum"),
                 pct_recovered=("recovered", "mean")).round(3).reset_index())
    print(desc.to_string(index=False))
    _save(desc, "02_instruction_descriptive")

    m = _fit_glmm_logit(zs, "recovered ~ C(instruction_type)")
    print("\n[GLMM logit: recovered ~ C(instruction_type) + (1|qid)]")
    print(m.summary().tables[1])
    _save(_coefs(m, "recovered_on_instruction"), "02_instruction_glmm")

    # Cochran's Q: paired by qid, ogni qid ha 4 istruzioni
    # Costruiamo wide qid x instruction su mean(recovered) over run
    wide = (zs.groupby(["qid", "instruction_type"])["recovered"]
            .mean().unstack("instruction_type"))
    # Cochran's Q richiede valori 0/1, quindi binarizziamo (almeno una run recovered)
    wide_bin = (wide > 0).astype(int).dropna()
    from statsmodels.stats.contingency_tables import cochrans_q
    if not wide_bin.empty and wide_bin.shape[1] >= 2:
        q = cochrans_q(wide_bin.values)
        print(f"\n[Cochran's Q] stat={q.statistic:.3f} df={q.df} p={q.pvalue:.4f}  (n_qid={len(wide_bin)})")
        _save(pd.DataFrame([{"test": "cochran_q_instruction",
                             "statistic": q.statistic, "df": q.df,
                             "p": q.pvalue, "n_qid": len(wide_bin)}]),
              "02_instruction_cochranq")

    # pairwise McNemar tra coppie di istruzioni
    from statsmodels.stats.contingency_tables import mcnemar
    pairs = []
    instrs = wide_bin.columns.tolist()
    for i in range(len(instrs)):
        for j in range(i + 1, len(instrs)):
            a, b = instrs[i], instrs[j]
            tab = pd.crosstab(wide_bin[a], wide_bin[b])
            tab = tab.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            res = mcnemar(tab.values, exact=False, correction=True)
            pairs.append({"a": a, "b": b,
                          "n_a_only": int(tab.loc[1, 0]),
                          "n_b_only": int(tab.loc[0, 1]),
                          "chi2": res.statistic, "p": res.pvalue})
    pdf = pd.DataFrame(pairs)
    if not pdf.empty:
        pdf["p_holm"] = multipletests(pdf["p"], method="holm")[1]
        print("\n[McNemar pairwise tra istruzioni (paired per qid)]")
        print(pdf.round(4).to_string(index=False))
        _save(pdf, "02_instruction_mcnemar_pairwise")


# ---------- Test 3 - n_hop ----------

def test_hop(zs: pd.DataFrame) -> None:
    _section("Test 3 - effetto della COMPLESSITA' (n_hop)")
    print("\n[descrittiva]")
    desc = (zs.groupby("n_hop")
            .agg(n_zero_start=("recovered", "size"),
                 n_recovered=("recovered", "sum"),
                 pct_recovered=("recovered", "mean")).round(3).reset_index())
    print(desc.to_string(index=False))
    _save(desc, "03_hop_descriptive")

    m = _fit_glmm_logit(zs, "recovered ~ C(n_hop)")
    print("\n[GLMM logit: recovered ~ C(n_hop) + (1|qid)]")
    print(m.summary().tables[1])
    _save(_coefs(m, "recovered_on_hop"), "03_hop_glmm")

    # Cochran-Armitage trend test (recovered vs hop ordinale)
    tab = pd.crosstab(zs["n_hop"], zs["recovered"]).reindex(index=[2, 3, 4])
    # implementazione: usa scipy.stats.chi2_contingency con trend score
    # statsmodels ha Linear-by-linear via ContingencyTable
    from statsmodels.stats.contingency_tables import Table2x2
    # Cochran-Armitage manuale con scores
    obs = tab.values
    n = obs.sum()
    scores = np.array([2, 3, 4], dtype=float)
    n_i_dot = obs.sum(axis=1)
    n_dot_1 = obs[:, 1].sum()
    p_dot_1 = n_dot_1 / n
    T = (obs[:, 1] - n_i_dot * p_dot_1) * scores
    T_num = T.sum()
    var = p_dot_1 * (1 - p_dot_1) * (n * (scores**2 * n_i_dot).sum() - (scores * n_i_dot).sum()**2) / n
    z = T_num / np.sqrt(var)
    p_trend = 2 * (1 - st.norm.cdf(abs(z)))
    print(f"\n[Cochran-Armitage trend test su n_hop] z={z:.3f} p={p_trend:.4f}")
    _save(pd.DataFrame([{"test": "cochran_armitage_hop",
                         "z": z, "p_two_sided": p_trend}]),
          "03_hop_trend")

    # chi2 di sanity 3x2
    chi2, p, dof, _ = st.chi2_contingency(tab)
    print(f"[sanity chi2 3x2] chi2={chi2:.3f} dof={dof} p={p:.4f}")
    _save(pd.DataFrame([{"test": "chi2_3x2_hop", "chi2": chi2, "dof": dof, "p": p}]),
          "03_hop_chi2")


# ---------- Test 4 - interazione hop x instruction ----------

def test_interaction(zs: pd.DataFrame) -> None:
    _section("Test 4 - interazione HOP x INSTRUCTION")
    print("\n[descrittiva: % recovered per hop x instruction]")
    pct = (zs.groupby(["n_hop", "instruction_type"])["recovered"]
           .mean().unstack("instruction_type").round(3))
    print(pct)
    pct_out = pct.reset_index()
    _save(pct_out, "04_interaction_descriptive")

    m_main = _fit_glmm_logit(zs, "recovered ~ C(n_hop) + C(instruction_type)")
    m_int = _fit_glmm_logit(zs, "recovered ~ C(n_hop) * C(instruction_type)")
    lr = 2 * (m_int.llf - m_main.llf)
    df_diff = len(m_int.params) - len(m_main.params)
    p_int = 1 - st.chi2.cdf(max(lr, 0), df=df_diff) if df_diff > 0 else float("nan")
    print(f"\n[LRT main vs interaction] chi2={lr:.3f} df={df_diff} p={p_int:.4f}")
    _save(pd.DataFrame([{"test": "LRT_interaction_hop_x_instruction",
                         "chi2": lr, "df": df_diff, "p": p_int}]),
          "04_interaction_lrt")
    _save(_coefs(m_int, "recovered_hop_x_instr"), "04_interaction_glmm")


# ---------- Test 5 - persistenza & step di recupero ----------

def test_persistence(zs: pd.DataFrame) -> None:
    _section("Test 5 - persistenza & step di primo recupero")
    rec = zs[zs["recovered"] == 1].copy()
    print(f"\n(analisi su n={len(rec)} chains recovered)")

    print("\n[n_positive_steps per instruction]")
    persist = (rec.groupby(["instruction_type", "n_pos_steps"]).size()
               .unstack("n_pos_steps", fill_value=0))
    print(persist)
    _save(persist.reset_index(), "05_persistence_by_instruction")

    # chi2 su persistenza x istruzione
    chi2, p, dof, _ = st.chi2_contingency(persist.values)
    print(f"[chi2 persistenza x instruction] chi2={chi2:.3f} dof={dof} p={p:.4f}")

    print("\n[step di primo recupero per instruction]")
    first = (rec.groupby(["instruction_type", "first_recovery_step"]).size()
             .unstack("first_recovery_step", fill_value=0))
    print(first)
    _save(first.reset_index(), "05_first_step_by_instruction")
    chi2_f, p_f, dof_f, _ = st.chi2_contingency(first.values)
    print(f"[chi2 step-di-recupero x instruction] chi2={chi2_f:.3f} dof={dof_f} p={p_f:.4f}")

    print("\n[step di primo recupero per hop]")
    first_h = (rec.groupby(["n_hop", "first_recovery_step"]).size()
               .unstack("first_recovery_step", fill_value=0))
    print(first_h)
    _save(first_h.reset_index(), "05_first_step_by_hop")
    chi2_h, p_h, dof_h, _ = st.chi2_contingency(first_h.values)
    print(f"[chi2 step-di-recupero x hop] chi2={chi2_h:.3f} dof={dof_h} p={p_h:.4f}")

    _save(pd.DataFrame([
        {"test": "chi2_persistence_by_instruction", "chi2": chi2, "dof": dof, "p": p},
        {"test": "chi2_first_step_by_instruction", "chi2": chi2_f, "dof": dof_f, "p": p_f},
        {"test": "chi2_first_step_by_hop", "chi2": chi2_h, "dof": dof_h, "p": p_h},
    ]), "05_chi2_summary")


# ---------- main ----------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _section("recovery analysis - 300q")
    zs = prepare(F1_CSV)
    print(f"\nzero-start chains: {len(zs)} / 3564 = {len(zs)/3564*100:.1f}%")
    print(f"of which recovered: {zs['recovered'].sum()} ({zs['recovered'].mean()*100:.1f}%)")

    test_group(zs)
    test_instruction(zs)
    test_hop(zs)
    test_interaction(zs)
    test_persistence(zs)


if __name__ == "__main__":
    main()
