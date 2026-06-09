"""BLEURT evaluation — dataset-agnostic.

Mirrors scripts/llama70b/_common/bleurt_eval.py. Three modes:
  - Baseline:    bleurt(E_t, E_0)
  - Consecutive: bleurt(E_t, E_{t-1})
  - Answer:      bleurt(predicted, gold)  (requires --f1-csv)

Uses BLEURT-20 by default (`pip install evaluate` + the BLEURT repo).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gptoss120b"))
from _common.olmo_constants import CHAIN_KEYS  # noqa: E402

DEFAULT_CHECKPOINT = os.environ.get("BLEURT_CHECKPOINT", "BLEURT-20")


def load_bleurt(checkpoint):
    import evaluate
    print(f"Loading BLEURT checkpoint: {checkpoint}")
    return evaluate.load("bleurt", checkpoint, module_type="metric")


def score_pairs(metric, cands, refs, batch_size, label):
    if not cands:
        return []
    print(f"  [{label}] scoring {len(cands)} pairs ...")
    t0 = time.time()
    scores = []
    for i in range(0, len(cands), batch_size):
        result = metric.compute(predictions=cands[i:i+batch_size],
                                references=refs[i:i+batch_size])
        scores.extend(result["scores"])
    print(f"  [{label}] done in {time.time() - t0:.1f}s")
    return scores


def build_step_index(df):
    return {
        (r["qid"], r["group"], r["instruction_type"], int(r["run"]), int(r["step"])): r["text"]
        for _, r in df.iterrows()
    }


def main():
    ap = argparse.ArgumentParser(description="BLEURT on OLMo-32B rewriting chains.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--f1-csv", type=Path, default=None,
                    help="Answer F1 CSV for the bleurt_answer mode (predicted vs gold).")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--no-consecutive", action="store_true")
    ap.add_argument("--no-answer", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    out_path = args.output or args.input.with_name(args.input.stem + "_bleurt.csv")
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    do_text = not (args.no_baseline and args.no_consecutive)
    do_answer = not args.no_answer and args.f1_csv is not None

    print("=" * 70)
    print(f"BLEURT — checkpoint={args.checkpoint}, batch={args.batch_size}")
    print(f"  input : {args.input}")
    print(f"  output: {out_path}")
    print(f"  baseline={not args.no_baseline}, consecutive={not args.no_consecutive}, answer={do_answer}")
    print("=" * 70)

    df = pd.read_csv(args.input)
    if args.smoke_test:
        first_qid = df["qid"].iloc[0]
        df = df[(df["qid"] == first_qid) & (df["run"] == 0)]
    target = df[df["step"] > 0].reset_index(drop=True)

    if out_path.exists() and not args.smoke_test:
        done = pd.read_csv(out_path)
        done_keys = set(zip(done["qid"], done["group"], done["instruction_type"], done["run"], done["step"]))
        target = target[~target.apply(
            lambda r: (r["qid"], r["group"], r["instruction_type"], int(r["run"]), int(r["step"])) in done_keys,
            axis=1,
        )].reset_index(drop=True)
        print(f"  resume: {len(done_keys)} done, {len(target)} remaining")

    if target.empty:
        print("Nothing to compute."); return

    metric = load_bleurt(args.checkpoint)

    bleurt_baseline = [None] * len(target)
    bleurt_consecutive = [None] * len(target)

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
                cands_b.append(et); refs_b.append(e0); idx_b.append(i)
            if et and eprev and not args.no_consecutive:
                cands_c.append(et); refs_c.append(eprev); idx_c.append(i)
        if cands_b:
            for k, i in enumerate(score_pairs(metric, cands_b, refs_b, args.batch_size, "Baseline")):
                bleurt_baseline[idx_b[k]] = i
        if cands_c:
            for k, i in enumerate(score_pairs(metric, cands_c, refs_c, args.batch_size, "Consecutive")):
                bleurt_consecutive[idx_c[k]] = i

    bleurt_answer = [None] * len(target)
    if do_answer:
        print(f"\nLoading F1 CSV: {args.f1_csv}")
        f1_df = pd.read_csv(args.f1_csv)
        if args.smoke_test:
            first_qid = df["qid"].iloc[0]
            f1_df = f1_df[(f1_df["qid"] == first_qid) & (f1_df["run"] == 0)]
        f1_index = {
            (r["qid"], r["group"], r["instruction_type"], int(r["run"]), int(r["step"])): (
                str(r["predicted_answer"]), str(r["gold_answer"])
            )
            for _, r in f1_df.iterrows()
        }
        cands_a, refs_a, idx_a = [], [], []
        for i, row in target.iterrows():
            key = (row["qid"], row["group"], row["instruction_type"], int(row["run"]), int(row["step"]))
            entry = f1_index.get(key)
            if entry:
                pred, gold = entry
                if pred and gold and pred.lower() not in ("nan", "none", ""):
                    cands_a.append(pred); refs_a.append(gold); idx_a.append(i)
        if cands_a:
            for k, i in enumerate(score_pairs(metric, cands_a, refs_a, args.batch_size, "Answer")):
                bleurt_answer[idx_a[k]] = i

    out = target[CHAIN_KEYS + ["step"]].copy()
    out["bleurt_baseline"] = bleurt_baseline
    out["bleurt_consecutive"] = bleurt_consecutive
    out["bleurt_answer"] = bleurt_answer

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.smoke_test:
        prev = pd.read_csv(out_path)
        merged = pd.concat([prev, out], ignore_index=True)
        merged = merged.drop_duplicates(subset=CHAIN_KEYS + ["step"], keep="last")
        merged.to_csv(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    for col, label in [("bleurt_baseline", "Baseline"),
                       ("bleurt_consecutive", "Consecutive"),
                       ("bleurt_answer", "Answer")]:
        if out[col].notna().any():
            print(f"\n{label}:")
            print(out.pivot_table(index="instruction_type", columns="step", values=col, aggfunc="mean").round(3))


if __name__ == "__main__":
    main()
