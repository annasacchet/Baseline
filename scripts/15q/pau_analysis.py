"""
P/A/U analysis (Laban et al., 2024) — RQ evaluation framework.

For each metric at each step, across stochastic repetitions:
  P — Performance   : mean across repetitions
  A — Aptitude      : 90th percentile
  U — Unreliability : 90th percentile - 10th percentile

The stochastic dimension is `repetition` (from rewriting_chains_pau.csv),
NOT `run` (which indexes the 3 wordings of each instruction).

Degradation patterns detected per (group, instruction_type):
  Capability erosion  : A decreases over steps
  Stability loss      : A stable, U increases
  Combined degradation: P down, A down, U up

Metrics:
  - factscore           (step 1-3 only; step 0 = NaN by design)
  - bert_f1_baseline    (semantic drift from E0, step 1-3)
  - bert_f1_consecutive (change from previous step, step 1-3)
  - n_tokens            (all steps)

Input CSVs (results/15q/):
  rewriting_chains_pau.csv              — main chains with `repetition` column
  rewriting_chains_pau_bertscore.csv    — BERTScore
  rewriting_chains_pau_openfactscore.csv — FactScore

Output:
  results/15q/pau_statistics.csv   — P/A/U per metric/group/instruction_type/step
  results/15q/pau_patterns.csv     — degradation pattern per metric/group/instruction_type
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "15q"

METRICS = ["factscore", "bert_f1_baseline", "bert_f1_consecutive", "n_tokens"]

GROUP_KEYS = ["group", "instruction_type"]

# Stochastic dimension: repetition index within same (qid, instruction, wording)
STOCHASTIC_DIM = "repetition"


def load_data(results_dir: Path) -> pd.DataFrame:
    main_csv = results_dir / "rewriting_chains_pau.csv"
    bert_csv = results_dir / "rewriting_chains_pau_bertscore.csv"
    fs_csv = results_dir / "rewriting_chains_pau_openfactscore.csv"

    for p in [main_csv, bert_csv, fs_csv]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p.name} not found. Run rewriting_pipeline_pau.py first, "
                "then the bertscore and openfactscore eval scripts on its output."
            )

    df_main = pd.read_csv(main_csv)
    df_bert = pd.read_csv(bert_csv)
    df_fs = pd.read_csv(fs_csv)

    merge_keys = ["qid", "group", "instruction_type", "run", STOCHASTIC_DIM, "step"]

    df = (
        df_main[merge_keys + ["n_tokens"]]
        .merge(
            df_bert[merge_keys + ["bert_f1_baseline", "bert_f1_consecutive"]],
            on=merge_keys,
            how="left",
        )
        .merge(
            df_fs[merge_keys + ["factscore"]],
            on=merge_keys,
            how="left",
        )
    )
    return df


def compute_pau(df: pd.DataFrame) -> pd.DataFrame:
    """Compute P/A/U per metric, group, instruction_type, step.

    Groups over all (qid, run, repetition) combinations so that both
    question variability and stochastic sampling contribute to the distribution,
    as intended by Laban et al. (2024).
    """
    records = []
    for (group, instr_type, step), grp in df.groupby(GROUP_KEYS + ["step"]):
        for metric in METRICS:
            vals = grp[metric].dropna()
            if vals.empty:
                continue
            p = vals.mean()
            a = np.percentile(vals, 90)
            u = np.percentile(vals, 90) - np.percentile(vals, 10)
            records.append({
                "group": group,
                "instruction_type": instr_type,
                "step": step,
                "metric": metric,
                "P": round(p, 4),
                "A": round(a, 4),
                "U": round(u, 4),
                "n_obs": len(vals),
            })
    return pd.DataFrame(records)


def detect_patterns(pau: pd.DataFrame) -> pd.DataFrame:
    """Classify degradation pattern per (group, instruction_type, metric)."""
    records = []
    for (group, instr_type, metric), grp in pau.groupby(GROUP_KEYS + ["metric"]):
        grp = grp.sort_values("step")
        steps = grp["step"].tolist()
        if len(steps) < 2:
            continue

        first = grp.iloc[0]
        last = grp.iloc[-1]

        a_down = last["A"] < first["A"]
        u_up = last["U"] > first["U"]
        p_down = last["P"] < first["P"]

        if a_down and p_down and u_up:
            pattern = "combined_degradation"
        elif a_down and not u_up:
            pattern = "capability_erosion"
        elif not a_down and u_up:
            pattern = "stability_loss"
        else:
            pattern = "stable"

        records.append({
            "group": group,
            "instruction_type": instr_type,
            "metric": metric,
            "steps_range": f"{steps[0]}-{steps[-1]}",
            "P_start": first["P"],
            "P_end": last["P"],
            "A_start": first["A"],
            "A_end": last["A"],
            "U_start": first["U"],
            "U_end": last["U"],
            "pattern": pattern,
        })
    return pd.DataFrame(records)


def print_summary(pau: pd.DataFrame, patterns: pd.DataFrame):
    print("\n=== P/A/U Statistics ===")
    for metric in METRICS:
        sub = pau[pau["metric"] == metric]
        if sub.empty:
            continue
        print(f"\n--- {metric} ---")
        pivot = sub.pivot_table(
            index=GROUP_KEYS,
            columns="step",
            values=["P", "A", "U"],
        )
        print(pivot.to_string())

    print("\n=== Degradation Patterns ===")
    for metric in METRICS:
        sub = patterns[patterns["metric"] == metric]
        if sub.empty:
            continue
        print(f"\n--- {metric} ---")
        print(sub[GROUP_KEYS + ["pattern", "A_start", "A_end", "U_start", "U_end"]].to_string(index=False))

    print("\n=== Pattern counts (all metrics) ===")
    print(patterns["pattern"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(description="P/A/U analysis on PAU rewriting chains.")
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help=f"Directory containing the PAU CSVs (default: {RESULTS_DIR}).",
    )
    args = parser.parse_args()

    df = load_data(args.results_dir)
    n_reps = df[STOCHASTIC_DIM].nunique()
    print(f"Loaded {len(df)} rows | steps: {sorted(df['step'].unique())} | repetitions: {n_reps}")

    pau = compute_pau(df)
    patterns = detect_patterns(pau)

    out_pau = args.results_dir / "pau_statistics.csv"
    out_pat = args.results_dir / "pau_patterns.csv"
    pau.to_csv(out_pau, index=False)
    patterns.to_csv(out_pat, index=False)
    print(f"\nSaved: {out_pau}")
    print(f"Saved: {out_pat}")

    print_summary(pau, patterns)


if __name__ == "__main__":
    main()
