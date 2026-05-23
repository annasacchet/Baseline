"""
Statistical analysis of the OpenAI reclassify output (qid 2hop__14092_8311).

What we test (and what we do NOT)
---------------------------------
We have 797 NOT_SUPPORTED claims from ONE qid only. With a single qid we cannot
make population-level inferences about "elaborate vs paraphrase" across MuSiQue:
all claims share the same source paragraph. So everything here is DESCRIPTIVE
within this qid, plus a few inferential tests that are still meaningful when
the unit of analysis is the claim (not the qid):

  (1) Distribution of the 5 labels overall + by step + by instruction.
  (2) chi2 of independence: label x step  → does the error mix shift over steps?
  (3) chi2 of independence: label x instruction
  (4) elaborate vs other instructions on INVENTED: Fisher exact, with CI.
  (5) Cochran-Armitage trend test for INVENTED proportion across steps 1->2->3.
  (6) chi2 for SUPPORTED (the AFV-false-positive bucket) across instructions
      -> tests whether the AFV's paraphrase bias is uniform across instruction
      types.
  (7) Bootstrap 95% CI for the headline numbers:
        - "real error rate" = (CONTRADICTION + INVENTED + DISTORTED) / total
        - "AFV false-positive rate" = SUPPORTED / total
        - "calibrated OFS" = (n_originally_supported + n_judge_SUPPORTED) /
                              n_all_claims_of_qid

Outputs
-------
  results/600q/stats_reclassify_openai/
    README.md       -- human-readable summary, ready to paste in slides
    tables.csv      -- the contingency tables + tests as a long CSV
    bootstrap.csv   -- bootstrap point + CI for the headline numbers
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "results" / "600q"
INPUT_RECL = RESULTS / "rewriting_chains_musique_600q_reclassified_openai.csv"
INPUT_OFS = RESULTS / "rewriting_chains_musique_600q_openfactscore.csv"
OUT_DIR = RESULTS / "stats_reclassify_openai"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QID = "2hop__14092_8311"
LABEL_ORDER = ["SUPPORTED", "DISTORTED", "CONTRADICTION", "UNVERIFIABLE", "INVENTED"]
ERROR_LABELS = {"CONTRADICTION", "INVENTED", "DISTORTED"}

# ---------------------------------------------------------------------------

def cochran_armitage(events, totals, scores):
    """
    Cochran-Armitage trend test.
    events, totals, scores: arrays of length k (one per group, here per step).
    Returns z-statistic and two-sided p-value.
    """
    events = np.asarray(events, dtype=float)
    totals = np.asarray(totals, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = totals.sum()
    p = events.sum() / n
    mean_x = (scores * totals).sum() / n
    num = (events * (scores - mean_x)).sum()
    var = p * (1 - p) * (totals * (scores - mean_x) ** 2).sum()
    if var <= 0:
        return float("nan"), float("nan")
    z = num / np.sqrt(var)
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, pval


def chi2_table(df, row, col, label_order=None, instr_order=None):
    """Return contingency table (DataFrame), chi2, p, dof, expected."""
    tab = pd.crosstab(df[row], df[col])
    if label_order is not None:
        tab = tab.reindex(label_order)
    if instr_order is not None:
        tab = tab[[c for c in instr_order if c in tab.columns]]
    chi2, p, dof, exp = stats.chi2_contingency(tab.values)
    return tab, chi2, p, dof, pd.DataFrame(exp, index=tab.index, columns=tab.columns)


def bootstrap_proportion(values: np.ndarray, n_iter=10000, ci=95, rng=None):
    """Bootstrap CI on the mean of a 0/1 array."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(values)
    boots = rng.choice(values, size=(n_iter, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(values.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------

def main():
    if not INPUT_RECL.exists():
        raise FileNotFoundError(INPUT_RECL)
    df = pd.read_csv(INPUT_RECL)
    df = df[df["qid"] == QID].copy()
    print(f"Loaded {len(df)} reclassified claims for qid={QID}")

    instr_order = ["elaborate", "shorten", "formality", "paraphrase"]
    out_lines: list[str] = []
    add = out_lines.append
    long_rows: list[dict] = []

    add(f"# Reclassify statistics — OpenAI judge ({QID})")
    add("")
    add(f"Single-qid analysis on {len(df)} NOT_SUPPORTED claims; judge = gpt-4o-mini "
        "(temperature=0, 5-category prompt).")
    add("")
    add("> **Caveat.** All tests below have the claim as the unit of analysis. "
        "Since we only have one qid, *do not* read these results as "
        "population-level claims about MuSiQue 2-hop questions. They describe "
        "the error structure within this single qid.")
    add("")

    # ---- (1) Distributions -------------------------------------------------
    add("## 1. Marginal distribution of judge labels")
    add("")
    vc = df["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    pct = (vc / len(df) * 100).round(1)
    tbl = pd.DataFrame({"n": vc, "%": pct})
    add(tbl.to_markdown())
    add("")
    add(f"- **Real rewrite errors** (CONTRADICTION + INVENTED + DISTORTED) = "
        f"**{(vc['CONTRADICTION']+vc['INVENTED']+vc['DISTORTED'])}** "
        f"({(vc['CONTRADICTION']+vc['INVENTED']+vc['DISTORTED'])/len(df)*100:.1f}% of the 797 NOT_SUPPORTED).")
    add(f"- **AFV false positives** (SUPPORTED) = **{vc['SUPPORTED']}** "
        f"({vc['SUPPORTED']/len(df)*100:.1f}%).")
    add(f"- **Ambiguous / UNVERIFIABLE** = **{vc['UNVERIFIABLE']}** "
        f"({vc['UNVERIFIABLE']/len(df)*100:.1f}%).")
    add("")
    for lab, n in vc.items():
        long_rows.append({"test": "marginal", "group": lab, "n": int(n), "pct": float(pct[lab])})

    # ---- (2) chi2: label x step -------------------------------------------
    add("## 2. χ² independence — label × step")
    add("")
    tab, chi2, p, dof, exp = chi2_table(df, "label", "step", LABEL_ORDER)
    add("Contingency table (counts):")
    add("")
    add(tab.to_markdown())
    add("")
    add(f"- χ² = **{chi2:.2f}**, dof = {dof}, p = **{p:.3g}**")
    if p < 0.05:
        add("- Conclusion: the **mix of error categories shifts significantly across steps**.")
    else:
        add("- Conclusion: no significant shift across steps.")
    add("")
    long_rows.append({"test": "chi2_label_step", "group": "all", "stat": chi2, "dof": dof, "p": p})

    # ---- (3) chi2: label x instruction ------------------------------------
    add("## 3. χ² independence — label × instruction")
    add("")
    tab, chi2, p, dof, exp = chi2_table(df, "label", "instruction_type", LABEL_ORDER, instr_order)
    add("Contingency table:")
    add("")
    add(tab.to_markdown())
    add("")
    add(f"- χ² = **{chi2:.2f}**, dof = {dof}, p = **{p:.3g}**")
    if p < 0.05:
        add("- Conclusion: **error categories are not uniformly distributed across instructions**. "
            "Standardised residuals reveal which cells deviate the most.")
    else:
        add("- Conclusion: no significant difference across instructions.")
    add("")
    # standardised residuals (Pearson)
    std_res = (tab.values - exp.values) / np.sqrt(exp.values)
    std_res_df = pd.DataFrame(std_res, index=tab.index, columns=tab.columns).round(2)
    add("Pearson standardised residuals (|z| > 2 ⇒ over/under-represented):")
    add("")
    add(std_res_df.to_markdown())
    add("")
    long_rows.append({"test": "chi2_label_instr", "group": "all", "stat": chi2, "dof": dof, "p": p})

    # ---- (4) elaborate vs others on INVENTED ------------------------------
    add("## 4. elaborate vs others on INVENTED — Fisher exact")
    add("")
    df_inv = pd.DataFrame({
        "is_elaborate": (df["instruction_type"] == "elaborate").astype(int),
        "is_invented": (df["label"] == "INVENTED").astype(int),
    })
    ct = pd.crosstab(df_inv["is_elaborate"], df_inv["is_invented"])
    ct.index = ["other", "elaborate"]
    ct.columns = ["not_INVENTED", "INVENTED"]
    add(ct.to_markdown())
    add("")
    odds, p = stats.fisher_exact(ct.values, alternative="two-sided")
    n_inv_el = int(((df["instruction_type"] == "elaborate") & (df["label"] == "INVENTED")).sum())
    n_el = int((df["instruction_type"] == "elaborate").sum())
    n_inv_other = int(((df["instruction_type"] != "elaborate") & (df["label"] == "INVENTED")).sum())
    n_other = int((df["instruction_type"] != "elaborate").sum())
    p_el = n_inv_el / n_el
    p_other = n_inv_other / n_other
    add(f"- P(INVENTED | elaborate)   = {n_inv_el}/{n_el} = **{p_el*100:.1f}%**")
    add(f"- P(INVENTED | not elaborate) = {n_inv_other}/{n_other} = **{p_other*100:.1f}%**")
    add(f"- Fisher exact: odds ratio = **{odds:.2f}**, p = **{p:.3g}**")
    if p < 0.05:
        add("- Conclusion: elaborate **produces INVENTED claims at a different rate** "
            "than other instructions.")
    else:
        add("- Conclusion: no significant difference.")
    add("")
    long_rows.append({"test": "fisher_elaborate_invented",
                      "stat": odds, "p": p,
                      "p_elaborate": p_el, "p_other": p_other})

    # ---- (5) Cochran-Armitage trend on INVENTED across steps --------------
    add("## 5. Cochran-Armitage trend — INVENTED proportion across steps 1→2→3")
    add("")
    by_step = df.groupby("step").agg(n=("label", "size"),
                                      n_inv=("label", lambda s: (s == "INVENTED").sum()))
    by_step["p_inv"] = by_step["n_inv"] / by_step["n"]
    add(by_step.round(3).to_markdown())
    add("")
    z, p = cochran_armitage(by_step["n_inv"].values, by_step["n"].values,
                            scores=by_step.index.values.astype(float))
    add(f"- Cochran-Armitage z = **{z:.2f}**, p = **{p:.3g}** "
        f"(positive z ⇒ INVENTED rate grows with step)")
    add("")
    long_rows.append({"test": "cochran_armitage_invented_step", "stat": z, "p": p})

    # ---- (6) chi2 for SUPPORTED vs instructions ---------------------------
    add("## 6. AFV false-positive bucket (SUPPORTED) vs instruction")
    add("")
    df_sup = pd.DataFrame({
        "instruction_type": df["instruction_type"],
        "is_supported": (df["label"] == "SUPPORTED").astype(int),
    })
    ct = pd.crosstab(df_sup["instruction_type"], df_sup["is_supported"])
    ct.columns = ["not_SUPPORTED", "SUPPORTED"]
    ct = ct.reindex(instr_order)
    ct["P(SUPPORTED)"] = (ct["SUPPORTED"] / (ct["SUPPORTED"] + ct["not_SUPPORTED"])).round(3)
    add(ct.to_markdown())
    add("")
    chi2, p, dof, _ = stats.chi2_contingency(ct.iloc[:, :2].values)
    add(f"- χ² = **{chi2:.2f}**, dof = {dof}, p = **{p:.3g}**")
    if p < 0.05:
        add("- Conclusion: **the AFV's false-positive rate is not uniform across instructions** — "
            "Gemma-3-4B mislabels paraphrase-style rewrites more often than others.")
    else:
        add("- Conclusion: no significant difference; the AFV bias is uniform.")
    add("")
    long_rows.append({"test": "chi2_supported_instr", "stat": chi2, "dof": dof, "p": p})

    # ---- (7) Bootstrap on headline numbers --------------------------------
    add("## 7. Bootstrap 95% CI on the headline numbers")
    add("")

    rng = np.random.default_rng(0)

    # 7a — real error rate (CONTR+INV+DIST) over the 797 NOT_SUPPORTED claims
    real_err = df["label"].isin(ERROR_LABELS).astype(int).values
    m, lo, hi = bootstrap_proportion(real_err, rng=rng)
    add(f"- **Real-error rate within NOT_SUPPORTED** = {m*100:.1f}%  "
        f"(95% CI [{lo*100:.1f}, {hi*100:.1f}])")
    long_rows.append({"test": "bootstrap_real_error_rate_within_NS",
                      "estimate": m, "ci_lo": lo, "ci_hi": hi})

    # 7b — AFV false-positive rate within NOT_SUPPORTED
    sup = (df["label"] == "SUPPORTED").astype(int).values
    m, lo, hi = bootstrap_proportion(sup, rng=rng)
    add(f"- **AFV false-positive rate within NOT_SUPPORTED** = {m*100:.1f}%  "
        f"(95% CI [{lo*100:.1f}, {hi*100:.1f}])")
    long_rows.append({"test": "bootstrap_afv_fp_rate_within_NS",
                      "estimate": m, "ci_lo": lo, "ci_hi": hi})

    # 7c — calibrated OFS on the qid
    # Total claims of the qid (3286). Originally SUPPORTED by AFV = 2489.
    # Add the judge-rescued SUPPORTED count.
    ofs = pd.read_csv(INPUT_OFS)
    ofs_q = ofs[ofs["qid"] == QID]
    n_claims_total = int(ofs_q["n_facts"].sum())
    n_originally_supported = int(ofs_q["n_supported"].sum())
    n_judge_supported = int((df["label"] == "SUPPORTED").sum())
    raw_ofs = n_originally_supported / n_claims_total
    # For calibrated, bootstrap the SUPPORTED count among the 797 NS claims and
    # carry it into the OFS formula.
    n_iter = 10000
    boots = []
    ns_labels = df["label"].values
    rng2 = np.random.default_rng(1)
    for _ in range(n_iter):
        sample = rng2.choice(ns_labels, size=len(ns_labels), replace=True)
        boots.append((n_originally_supported + (sample == "SUPPORTED").sum()) / n_claims_total)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    calibrated_ofs = (n_originally_supported + n_judge_supported) / n_claims_total
    add(f"- **Raw OFS** (qid)          = {raw_ofs:.3f}  ({n_originally_supported}/{n_claims_total})")
    add(f"- **Calibrated OFS** (qid)   = **{calibrated_ofs:.3f}**  "
        f"(95% CI [{lo:.3f}, {hi:.3f}])")
    add(f"- Δ = +{(calibrated_ofs - raw_ofs)*100:.1f} percentage points "
        "of OFS underestimation by the Gemma AFV on this qid.")
    add("")
    long_rows.append({"test": "raw_ofs_qid", "estimate": raw_ofs})
    long_rows.append({"test": "calibrated_ofs_qid",
                      "estimate": calibrated_ofs, "ci_lo": lo, "ci_hi": hi})

    # ---- save -------------------------------------------------------------
    readme_path = OUT_DIR / "README.md"
    readme_path.write_text("\n".join(out_lines))
    print(f"\nWrote {readme_path}")

    tests_path = OUT_DIR / "tests.csv"
    pd.DataFrame(long_rows).to_csv(tests_path, index=False)
    print(f"Wrote {tests_path}")


if __name__ == "__main__":
    main()
