"""
BLEURT evaluation for the 300q rewriting chains.

Three modes (all computed by default):
  - Baseline:    bleurt(E_t, E_0)       — semantic drift from original text
  - Consecutive: bleurt(E_t, E_{t-1})   — change between adjacent steps
  - Answer:      bleurt(predicted, gold) — semantic match of QA output vs gold
                 (requires --f1-csv; uses predicted_answer and gold_answer columns)

Output CSV (one row per step > 0):
  qid, group, instruction_type, run, step,
  bleurt_baseline, bleurt_consecutive,   [from text chain]
  bleurt_answer                           [from F1 CSV, NaN if not requested]

Requires:
  pip install evaluate
  # BLEURT checkpoint is downloaded automatically on first run (~1.6 GB for BLEURT-20)
  # or set BLEURT_CHECKPOINT to a local path

Uso:
  python scripts/300q/bleurt_eval_300q.py
  python scripts/300q/bleurt_eval_300q.py --f1-csv results/300q/rewriting_chains_300q_answer_f1_olmo31_4bit.csv
  python scripts/300q/bleurt_eval_300q.py --checkpoint BLEURT-20-D12  # smaller/faster
  python scripts/300q/bleurt_eval_300q.py --smoke-test
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "300q"

DEFAULT_CHAINS_CSV = RESULTS_DIR / "rewriting_chains_300q.csv"
DEFAULT_F1_CSV = RESULTS_DIR / "rewriting_chains_300q_answer_f1_olmo31_4bit.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "rewriting_chains_300q_bleurt.csv"

DEFAULT_CHECKPOINT = os.environ.get("BLEURT_CHECKPOINT", "BLEURT-20")

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]
SMOKE_QID = "2hop__635544_110949"


def load_bleurt(checkpoint: str):
    try:
        import evaluate
    except ImportError:
        raise ImportError(
            "Package 'evaluate' not found. Install with: pip install evaluate\n"
            "BLEURT checkpoint (~1.6 GB) will be downloaded automatically on first run."
        )
    print(f"Loading BLEURT checkpoint: {checkpoint}")
    metric = evaluate.load("bleurt", checkpoint, module_type="metric")
    print("  BLEURT ready.")
    return metric


def score_pairs(metric, candidates: list[str], references: list[str], batch_size: int, label: str) -> list[float]:
    if not candidates:
        return []
    print(f"  [{label}] scoring {len(candidates)} pairs...")
    t0 = time.time()
    scores: list[float] = []
    for i in range(0, len(candidates), batch_size):
        batch_c = candidates[i:i + batch_size]
        batch_r = references[i:i + batch_size]
        result = metric.compute(predictions=batch_c, references=batch_r)
        scores.extend(result["scores"])
        n_done = min(i + batch_size, len(candidates))
        elapsed = time.time() - t0
        eta = (len(candidates) - n_done) * elapsed / max(n_done, 1)
        print(f"    {n_done}/{len(candidates)}  ETA {eta/60:.1f} min", end="\r")
    print(f"\n  [{label}] done in {time.time() - t0:.1f}s")
    return scores


def build_step_index(df: pd.DataFrame) -> dict:
    return {
        (row["qid"], row["group"], row["instruction_type"], int(row["run"]), int(row["step"])): row["text"]
        for _, row in df.iterrows()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BLEURT evaluation on 300q rewriting chains.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CHAINS_CSV)
    parser.add_argument("--f1-csv", type=Path, default=None,
                        help="Answer F1 CSV for bleurt_answer mode. Defaults to the standard 300q F1 file if it exists.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help=f"BLEURT checkpoint name or local path (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline (E_t vs E_0) mode.")
    parser.add_argument("--no-consecutive", action="store_true", help="Skip consecutive (E_t vs E_{t-1}) mode.")
    parser.add_argument("--no-answer", action="store_true", help="Skip answer-level BLEURT mode.")
    parser.add_argument("--smoke-test", action="store_true", help=f"Run only on {SMOKE_QID}, run 0.")
    args = parser.parse_args()

    # Resolve F1 CSV default
    f1_csv_path: Path | None = args.f1_csv
    if f1_csv_path is None and not args.no_answer:
        f1_csv_path = DEFAULT_F1_CSV if DEFAULT_F1_CSV.exists() else None
        if f1_csv_path is None:
            print(f"[info] F1 CSV not found at default path, skipping bleurt_answer. "
                  f"Pass --f1-csv to enable it.")

    do_text = not (args.no_baseline and args.no_consecutive)
    do_answer = not args.no_answer and f1_csv_path is not None

    print("=" * 70)
    print(f"BLEURT eval — checkpoint={args.checkpoint}, batch={args.batch_size}")
    print(f"  baseline={not args.no_baseline}, consecutive={not args.no_consecutive}, answer={do_answer}")
    print("=" * 70)

    if not args.input.exists():
        raise FileNotFoundError(f"Chains CSV not found: {args.input}")

    print(f"\nLoading chains: {args.input}")
    df = pd.read_csv(args.input)
    if args.smoke_test:
        df = df[(df["qid"] == SMOKE_QID) & (df["run"] == 0)]
        print(f"*** SMOKE TEST: {len(df)} rows ***")
    print(f"  {len(df)} rows | {df.groupby(CHAIN_KEYS).ngroups} chains")

    target = df[df["step"] > 0].reset_index(drop=True)
    print(f"  rows to score (step > 0): {len(target)}")

    metric = load_bleurt(args.checkpoint)

    # -------------------------------------------------------------------------
    # Text-level modes: Baseline and Consecutive
    # -------------------------------------------------------------------------
    bleurt_baseline: list[float | None] = [None] * len(target)
    bleurt_consecutive: list[float | None] = [None] * len(target)

    if do_text:
        step_index = build_step_index(df)

        cands_b, refs_b, idx_b = [], [], []
        cands_c, refs_c, idx_c = [], [], []

        for i, row in target.iterrows():
            qid, group, itype, run, step = (
                row["qid"], row["group"], row["instruction_type"],
                int(row["run"]), int(row["step"]),
            )
            et = step_index.get((qid, group, itype, run, step))
            e0 = step_index.get((qid, group, itype, run, 0))
            eprev = step_index.get((qid, group, itype, run, step - 1))

            if et and e0 and not args.no_baseline:
                cands_b.append(et)
                refs_b.append(e0)
                idx_b.append(i)

            if et and eprev and not args.no_consecutive:
                cands_c.append(et)
                refs_c.append(eprev)
                idx_c.append(i)

        if cands_b:
            scores_b = score_pairs(metric, cands_b, refs_b, args.batch_size, "Baseline")
            for k, i in enumerate(idx_b):
                bleurt_baseline[i] = scores_b[k]

        if cands_c:
            scores_c = score_pairs(metric, cands_c, refs_c, args.batch_size, "Consecutive")
            for k, i in enumerate(idx_c):
                bleurt_consecutive[i] = scores_c[k]

    # -------------------------------------------------------------------------
    # Answer-level mode: bleurt(predicted_answer, gold_answer)
    # -------------------------------------------------------------------------
    bleurt_answer: list[float | None] = [None] * len(target)

    if do_answer:
        print(f"\nLoading F1 CSV: {f1_csv_path}")
        f1_df = pd.read_csv(f1_csv_path)
        if args.smoke_test:
            f1_df = f1_df[(f1_df["qid"] == SMOKE_QID) & (f1_df["run"] == 0)]

        f1_index = {
            (row["qid"], row["group"], row["instruction_type"], int(row["run"]), int(row["step"])): (
                str(row["predicted_answer"]), str(row["gold_answer"])
            )
            for _, row in f1_df.iterrows()
        }

        cands_a, refs_a, idx_a = [], [], []
        for i, row in target.iterrows():
            key = (row["qid"], row["group"], row["instruction_type"], int(row["run"]), int(row["step"]))
            entry = f1_index.get(key)
            if entry:
                pred, gold = entry
                if pred and gold and pred.lower() not in ("nan", "none", ""):
                    cands_a.append(pred)
                    refs_a.append(gold)
                    idx_a.append(i)

        if cands_a:
            scores_a = score_pairs(metric, cands_a, refs_a, args.batch_size, "Answer")
            for k, i in enumerate(idx_a):
                bleurt_answer[i] = scores_a[k]

    # -------------------------------------------------------------------------
    # Build output
    # -------------------------------------------------------------------------
    out = target[CHAIN_KEYS + ["step"]].copy()
    out["bleurt_baseline"] = bleurt_baseline
    out["bleurt_consecutive"] = bleurt_consecutive
    out["bleurt_answer"] = bleurt_answer

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.smoke_test:
        prev = pd.read_csv(args.output)
        merged = pd.concat([prev, out], ignore_index=True)
        merged = merged.drop_duplicates(subset=CHAIN_KEYS + ["step"], keep="last")
        merged.to_csv(args.output, index=False)
    else:
        out.to_csv(args.output, index=False)

    print(f"\nSaved: {args.output}")

    # -------------------------------------------------------------------------
    # Summary pivot
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BLEURT — mean per (instruction_type, step)")
    print("=" * 70)

    for col, label in [
        ("bleurt_baseline", "Baseline (E_t vs E_0)"),
        ("bleurt_consecutive", "Consecutive (E_t vs E_{t-1})"),
        ("bleurt_answer", "Answer (predicted vs gold)"),
    ]:
        if out[col].notna().any():
            print(f"\n{label}:")
            pivot = out.pivot_table(index="instruction_type", columns="step", values=col, aggfunc="mean")
            print(pivot.round(3))


if __name__ == "__main__":
    main()
