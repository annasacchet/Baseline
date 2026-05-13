"""
Analisi mirata dell'istruzione `elaborate`.

Risponde alla domanda: "cosa fa davvero elaborate?".
Quattro blocchi:
  1) Descrittivo: distribuzione di n_tokens, compliance (elabora davvero?)
  2) Contrasti: elaborate vs altre istruzioni su F1/OFS/BERTScore
  3) n_tokens come predittore: la lunghezza guida la fattualita?
  4) Failure mode: catene "corte" vs "lunghe" dentro elaborate

Funziona per il run 300q e per il pilot 15q.

Uso:
    python3.11 scripts/_common/elaborate_analysis.py --dataset 300q
    python3.11 scripts/_common/elaborate_analysis.py --dataset 15q
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "300q": {
        "chains": "results/300q/rewriting_chains_300q.csv",
        "f1":     "results/300q/rewriting_chains_300q_answer_f1_olmo31_4bit.csv",
        "ofs":    "results/300q/rewriting_chains_300q_openfactscore.csv",
        "bs":     "results/300q/rewriting_chains_300q_bertscore.csv",
        "out":    "results/300q/stats/elaborate",
    },
    "15q": {
        "chains": "results/15q/rewriting_chains_15q.csv",
        "f1":     "results/15q/rewriting_chains_15q_answer_f1.csv",
        "ofs":    "results/15q/rewriting_chains_15q_openfactscore.csv",
        "bs":     "results/15q/rewriting_chains_15q_bertscore.csv",
        "out":    "results/15q/stats/elaborate",
    },
}

KEYS = ["qid", "group", "instruction_type", "run", "step"]


def _section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def _save(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.csv"
    df.to_csv(p, index=False)
    print(f"  [saved] {p.relative_to(ROOT)}")


# ---------- caricamento ----------

def load(dataset: str) -> dict:
    cfg = DATASETS[dataset]
    chains = pd.read_csv(ROOT / cfg["chains"])
    f1 = pd.read_csv(ROOT / cfg["f1"])
    ofs = pd.read_csv(ROOT / cfg["ofs"])
    bs = pd.read_csv(ROOT / cfg["bs"])

    # lunghezza dell'originale per qid
    orig_tokens = (chains[chains["step"] == 0]
                   .groupby("qid")["n_tokens"].mean().rename("orig_tokens"))
    chains = chains.merge(orig_tokens, on="qid", how="left")

    # join chains <-> metriche (via KEYS)
    merged = (chains[KEYS + ["n_tokens", "orig_tokens"]]
              .merge(f1[KEYS + ["answer_f1"]], on=KEYS, how="left")
              .merge(ofs[KEYS + ["factscore"]], on=KEYS, how="left")
              .merge(bs[KEYS + ["bert_f1_baseline"]], on=KEYS, how="left"))

    # hop
    merged["n_hop"] = merged["qid"].str.extract(r"^(\d+)hop")[0].astype("Int64")
    merged["f1_bin"] = (merged["answer_f1"] > 0).astype("Int64")
    # ratio rispetto all'originale
    merged["tok_ratio"] = merged["n_tokens"] / merged["orig_tokens"]

    return {"all": merged,
            "elaborate": merged[merged["instruction_type"] == "elaborate"].copy(),
            "cfg": cfg}


# ---------- Blocco 1: descrittivo ----------

def block1_descriptive(d: dict, out_dir: Path) -> None:
    _section("Blocco 1 - descrittivo: cosa fa elaborate ai token")
    el = d["elaborate"]
    chains = d["all"]

    # 1.1 distribuzione n_tokens per step (elaborate)
    desc = el.groupby("step")["n_tokens"].describe().round(1)
    print("\n[elaborate: n_tokens per step]")
    print(desc)
    desc_out = desc.reset_index()
    _save(desc_out, out_dir, "01_elaborate_ntokens_by_step")

    # 1.2 compliance: % catene che AUMENTANO la lunghezza vs originale
    el = el.copy()
    el["increased"] = (el["n_tokens"] > el["orig_tokens"]).astype(int)
    el["halved"] = (el["n_tokens"] < 0.5 * el["orig_tokens"]).astype(int)
    el["catastrophic"] = (el["n_tokens"] < 200).astype(int)
    comp = (el[el["step"] >= 1]
            .groupby("step")
            .agg(n=("increased", "size"),
                 pct_increased=("increased", "mean"),
                 pct_halved=("halved", "mean"),
                 pct_catastrophic=("catastrophic", "mean"),
                 mean_ratio=("tok_ratio", "mean"),
                 median_ratio=("tok_ratio", "median"))
            .round(3).reset_index())
    print("\n[elaborate: compliance (% catene per step)]")
    print(comp.to_string(index=False))
    _save(comp, out_dir, "02_elaborate_compliance")

    # 1.3 confronto n_tokens elaborate vs altre istruzioni per step
    cross = (chains[chains["step"] >= 1]
             .groupby(["instruction_type", "step"])["n_tokens"]
             .agg(["mean", "median", "std"]).round(1).reset_index())
    print("\n[n_tokens per istruzione x step (tutte)]")
    pivot = cross.pivot(index="instruction_type", columns="step", values="mean")
    print(pivot)
    _save(cross, out_dir, "03_ntokens_by_instruction_step")

    # 1.4 Wilcoxon elaborate vs shorten (entrambe content): same n_tokens?
    rows = []
    for s in sorted(chains[chains["step"] >= 1]["step"].unique()):
        sub = chains[(chains["step"] == s) &
                     (chains["instruction_type"].isin(["elaborate", "shorten"]))]
        wide = (sub.groupby(["qid", "instruction_type"])["n_tokens"]
                .mean().unstack("instruction_type").dropna())
        if {"elaborate", "shorten"}.issubset(wide.columns):
            stat, p = st.wilcoxon(wide["elaborate"], wide["shorten"])
            rows.append({"step": int(s),
                         "n_pairs": len(wide),
                         "mean_elaborate": wide["elaborate"].mean(),
                         "mean_shorten": wide["shorten"].mean(),
                         "mean_diff": (wide["elaborate"] - wide["shorten"]).mean(),
                         "wilcoxon_stat": stat, "p": p})
    comp_es = pd.DataFrame(rows)
    if len(comp_es):
        comp_es["p_holm"] = multipletests(comp_es["p"], method="holm")[1]
        print("\n[elaborate vs shorten: n_tokens, Wilcoxon paired]")
        print(comp_es.round(4).to_string(index=False))
        _save(comp_es, out_dir, "04_elaborate_vs_shorten_ntokens")


# ---------- Blocco 2: elaborate vs altre istruzioni sulle metriche ----------

def block2_contrasts(d: dict, out_dir: Path) -> None:
    _section("Blocco 2 - elaborate vs altre istruzioni sulle metriche")
    df = d["all"]

    metric_cols = [("answer_f1", "F1"),
                   ("factscore", "OFS"),
                   ("bert_f1_baseline", "BS_base")]

    others = ["shorten", "formality", "paraphrase"]
    rows = []
    for col, label in metric_cols:
        sub_all = df.dropna(subset=[col, "instruction_type"])
        for other in others:
            sub = sub_all[(sub_all["instruction_type"].isin(["elaborate", other])) &
                          (sub_all["step"] >= 1)]
            if sub.empty:
                continue
            wide = (sub.groupby(["qid", "instruction_type"])[col]
                    .mean().unstack("instruction_type").dropna())
            if {"elaborate", other}.issubset(wide.columns):
                stat, p = st.wilcoxon(wide["elaborate"], wide[other],
                                      zero_method="zsplit")
                rows.append({"metric": label,
                             "contrast": f"elaborate vs {other}",
                             "n_pairs": len(wide),
                             "mean_elaborate": wide["elaborate"].mean(),
                             "mean_other": wide[other].mean(),
                             "mean_diff": (wide["elaborate"] - wide[other]).mean(),
                             "wilcoxon_stat": stat, "p": p})
    contrasts = pd.DataFrame(rows)
    if not contrasts.empty:
        contrasts["p_holm"] = multipletests(contrasts["p"].fillna(1),
                                             method="holm")[1]
    print(contrasts.round(4).to_string(index=False))
    _save(contrasts, out_dir, "10_elaborate_vs_other_metrics")


# ---------- Blocco 3: n_tokens come predittore (mediation) ----------

def _fit_lmm(df: pd.DataFrame, formula: str, group_col: str = "qid"):
    df = df.copy()
    df["_grp"] = df[group_col].astype("category")
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


def block3_predictor(d: dict, out_dir: Path) -> None:
    _section("Blocco 3 - n_tokens come predittore (dentro elaborate)")
    el = d["elaborate"]
    # restringi a step >= 1 (step 0 e' l'originale, OFS NA)
    el = el[el["step"] >= 1].copy()
    el["log_ntokens"] = np.log(el["n_tokens"].clip(lower=1))

    print("\n[F1 ~ log(n_tokens) + step + (1|qid)  - solo elaborate]")
    sub = el.dropna(subset=["answer_f1"]).copy()
    m_f1 = _fit_lmm(sub, "answer_f1 ~ log_ntokens + C(step)")
    print(m_f1.summary().tables[1])
    _save(_coefs(m_f1, "F1_on_logtokens_elaborate"),
          out_dir, "20_f1_on_logtokens_elaborate")

    print("\n[OFS ~ log(n_tokens) + step + (1|qid)  - solo elaborate]")
    sub = el.dropna(subset=["factscore"]).copy()
    if len(sub) > 5:
        m_ofs = _fit_lmm(sub, "factscore ~ log_ntokens + C(step)")
        print(m_ofs.summary().tables[1])
        _save(_coefs(m_ofs, "OFS_on_logtokens_elaborate"),
              out_dir, "21_ofs_on_logtokens_elaborate")
    else:
        print("  [skip] dati insufficienti")

    # mediation-style: instruction effect on OFS si attenua aggiungendo n_tokens?
    df = d["all"][d["all"]["step"] >= 1].copy()
    df["log_ntokens"] = np.log(df["n_tokens"].clip(lower=1))
    df = df.dropna(subset=["factscore", "log_ntokens"])
    print("\n[OFS ~ C(instruction_type) + C(step) + (1|qid)  - senza n_tokens]")
    m_a = _fit_lmm(df, "factscore ~ C(instruction_type) + C(step)")
    print(m_a.summary().tables[1])
    _save(_coefs(m_a, "OFS_on_instruction_no_tokens"),
          out_dir, "22_ofs_on_instruction_no_tokens")
    print("\n[OFS ~ C(instruction_type) + C(step) + log(n_tokens) + (1|qid)  - con n_tokens]")
    m_b = _fit_lmm(df, "factscore ~ C(instruction_type) + C(step) + log_ntokens")
    print(m_b.summary().tables[1])
    _save(_coefs(m_b, "OFS_on_instruction_with_tokens"),
          out_dir, "23_ofs_on_instruction_with_tokens")


# ---------- Blocco 4: failure mode ----------

def block4_failure(d: dict, out_dir: Path) -> None:
    _section("Blocco 4 - failure mode dentro elaborate")
    el = d["elaborate"].copy()
    el = el[el["step"] >= 1].copy()
    el["short"] = (el["n_tokens"] < 200).astype(int)

    metrics = [("answer_f1", "F1"),
               ("factscore", "OFS"),
               ("bert_f1_baseline", "BS_base")]
    rows = []
    for col, label in metrics:
        sub = el.dropna(subset=[col]).copy()
        if sub["short"].nunique() < 2:
            continue
        long_, short_ = sub[sub["short"] == 0][col], sub[sub["short"] == 1][col]
        try:
            stat, p = st.mannwhitneyu(short_, long_, alternative="two-sided")
        except ValueError:
            stat, p = float("nan"), float("nan")
        rows.append({"metric": label,
                     "n_short": len(short_), "n_long": len(long_),
                     "mean_short": short_.mean(), "mean_long": long_.mean(),
                     "mean_diff": short_.mean() - long_.mean(),
                     "mw_stat": stat, "p": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = multipletests(out["p"].fillna(1), method="holm")[1]
    print(out.round(4).to_string(index=False))
    _save(out, out_dir, "30_failure_short_vs_long")

    # caratterizzazione dei "short": per hop, per group, per run
    char = (el.groupby(["n_hop", "step"])["short"].mean()
            .unstack("step").round(3))
    print("\n[% catene 'short' (<200 tok) per n_hop x step]")
    print(char)
    char_out = char.reset_index()
    _save(char_out, out_dir, "31_short_rate_by_hop_step")

    char_g = (el.groupby(["group", "step"])["short"].mean()
              .unstack("step").round(3))
    print("\n[% catene 'short' per group x step]")
    print(char_g)
    _save(char_g.reset_index(), out_dir, "32_short_rate_by_group_step")


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    args = p.parse_args()

    out_dir = ROOT / DATASETS[args.dataset]["out"]
    _section(f"elaborate analysis - dataset = {args.dataset}")
    d = load(args.dataset)
    print(f"merged rows = {len(d['all'])}, elaborate rows = {len(d['elaborate'])}")

    block1_descriptive(d, out_dir)
    block2_contrasts(d, out_dir)
    block3_predictor(d, out_dir)
    block4_failure(d, out_dir)


if __name__ == "__main__":
    main()
