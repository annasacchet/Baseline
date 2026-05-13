"""
Statistiche aggregate per il run 300q.

Carica i CSV in results/300q/ e stampa medie/std/conteggi per:
- Answer F1 (OLMo-3.1 4-bit)
- OpenFActScore (4-bit)
- BERTScore (baseline vs step 0, consecutive vs step k-1)

Breakdown: per group, per n_hops, per instruction_type, per step,
e incroci n_hops x instruction_type x step.

Uso:
    python3.11 scripts/300q/metrics_stats.py
    python3.11 scripts/300q/metrics_stats.py --save        # salva CSV in results/300q/stats/
    python3.11 scripts/300q/metrics_stats.py --metric f1   # solo Answer F1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "300q"
STATS_DIR = RESULTS_DIR / "stats"

FILES = {
    "f1": RESULTS_DIR / "rewriting_chains_300q_answer_f1_olmo31_4bit.csv",
    "ofs": RESULTS_DIR / "rewriting_chains_300q_openfactscore.csv",
    "bertscore": RESULTS_DIR / "rewriting_chains_300q_bertscore.csv",
}

METRIC_COL = {
    "f1": "answer_f1",
    "ofs": "factscore",
    "bertscore_baseline": "bert_f1_baseline",
    "bertscore_consecutive": "bert_f1_consecutive",
}


def _add_hops(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_hops"] = df["qid"].str.extract(r"^(\d+)hop").astype(int)
    return df


def _agg(df: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    return (
        df.groupby(by)[metric]
        .agg(["mean", "std", "count"])
        .round(3)
        .reset_index()
    )


def _pivot(df: pd.DataFrame, row: str, metric: str) -> pd.DataFrame:
    return (
        df.groupby([row, "step"])[metric]
        .mean()
        .unstack("step")
        .round(3)
    )


def _section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def _report(df: pd.DataFrame, metric: str, label: str, save: bool) -> None:
    df = _add_hops(df)
    _section(f"{label}  (n={len(df)}, col={metric})")

    by_group = _agg(df, ["group"], metric)
    by_hops = _agg(df, ["n_hops"], metric)
    by_instr = _agg(df, ["instruction_type"], metric)
    by_step = _agg(df, ["step"], metric)
    pivot_hops = _pivot(df, "n_hops", metric)
    pivot_instr = _pivot(df, "instruction_type", metric)
    pivot_full = (
        df.groupby(["n_hops", "instruction_type", "step"])[metric]
        .mean()
        .unstack("step")
        .round(3)
    )

    print("\n-- per group --")
    print(by_group.to_string(index=False))
    print("\n-- per n_hops --")
    print(by_hops.to_string(index=False))
    print("\n-- per instruction_type --")
    print(by_instr.to_string(index=False))
    print("\n-- per step --")
    print(by_step.to_string(index=False))
    print("\n-- n_hops x step --")
    print(pivot_hops)
    print("\n-- instruction_type x step --")
    print(pivot_instr)
    print("\n-- n_hops x instruction_type x step --")
    print(pivot_full)

    if save:
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        prefix = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        by_group.to_csv(STATS_DIR / f"{prefix}_by_group.csv", index=False)
        by_hops.to_csv(STATS_DIR / f"{prefix}_by_hops.csv", index=False)
        by_instr.to_csv(STATS_DIR / f"{prefix}_by_instruction.csv", index=False)
        by_step.to_csv(STATS_DIR / f"{prefix}_by_step.csv", index=False)
        pivot_hops.to_csv(STATS_DIR / f"{prefix}_hops_x_step.csv")
        pivot_instr.to_csv(STATS_DIR / f"{prefix}_instruction_x_step.csv")
        pivot_full.to_csv(STATS_DIR / f"{prefix}_hops_x_instruction_x_step.csv")
        print(f"\n[saved] {STATS_DIR}/{prefix}_*.csv")


def run_f1(save: bool) -> None:
    df = pd.read_csv(FILES["f1"])
    _report(df, METRIC_COL["f1"], "Answer F1 (OLMo-3.1 4-bit)", save)


def run_ofs(save: bool) -> None:
    df = pd.read_csv(FILES["ofs"])
    _report(df, METRIC_COL["ofs"], "OpenFActScore (4-bit)", save)


def run_bertscore(save: bool) -> None:
    df = pd.read_csv(FILES["bertscore"])
    _report(df, METRIC_COL["bertscore_baseline"], "BERTScore F1 (vs step 0)", save)
    _report(df, METRIC_COL["bertscore_consecutive"], "BERTScore F1 (vs step k-1)", save)


RUNNERS = {
    "f1": run_f1,
    "ofs": run_ofs,
    "bertscore": run_bertscore,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metric",
        choices=["all", *RUNNERS.keys()],
        default="all",
        help="quale metrica calcolare (default: all)",
    )
    p.add_argument("--save", action="store_true", help="salva i breakdown in results/300q/stats/")
    args = p.parse_args()

    targets = list(RUNNERS) if args.metric == "all" else [args.metric]
    for t in targets:
        RUNNERS[t](args.save)


if __name__ == "__main__":
    main()
