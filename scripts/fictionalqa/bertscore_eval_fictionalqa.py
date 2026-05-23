"""
BERTScore evaluation for FictionalQA rewriting chains.

Mirrors scripts/newsqa/bertscore_eval_newsqa.py: two modes
  - Baseline:    sim(E_t, E_0)
  - Consecutive: sim(E_t, E_{t-1})

Model: roberta-large (layer 17).
Resumable via --output.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from bert_score import score as compute_bert_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = REPO_ROOT / "results" / "fictionalqa" / "rewriting_chains_fictionalqa.csv"
DEFAULT_OUT = REPO_ROOT / "results" / "fictionalqa" / "rewriting_chains_fictionalqa_bertscore.csv"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

BERT_MODEL = "roberta-large"
BERT_NUM_LAYERS = 17
BERT_LANG = "en"


def build_step_index(chains_df: pd.DataFrame) -> dict:
    idx = {}
    for _, row in chains_df.iterrows():
        key = (
            row["qid"], row["group"], row["instruction_type"],
            int(row["run"]), int(row["step"]),
        )
        idx[key] = row["text"]
    return idx


def compute_bertscore_pairs(candidates, references, batch_size, device, label):
    if not candidates:
        return [], [], []
    print(f"  [{label}] computing BERTScore on {len(candidates)} pairs...")
    t0 = time.time()
    P, R, F1 = compute_bert_score(
        candidates,
        references,
        lang=BERT_LANG,
        model_type=BERT_MODEL,
        num_layers=BERT_NUM_LAYERS,
        batch_size=batch_size,
        device=device,
        verbose=True,
    )
    dt = time.time() - t0
    print(f"  [{label}] done in {dt:.1f}s ({len(candidates)/dt:.1f} pairs/s)")
    return P.tolist(), R.tolist(), F1.tolist()


def main():
    parser = argparse.ArgumentParser(description="BERTScore (Baseline + Consecutive) on FictionalQA chains CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 70)
    print(f"BERTScore FictionalQA  —  device={device}, batch={args.batch_size}")
    print(f"  model: {BERT_MODEL} (layer {BERT_NUM_LAYERS})")
    print("=" * 70)

    if not args.input.exists():
        raise FileNotFoundError(f"Chain CSV not found: {args.input}")

    print(f"\nLoading chains: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  rows: {len(df)}  |  unique chains: {df.groupby(CHAIN_KEYS).ngroups}")

    step_index = build_step_index(df)

    target = df[df["step"] > 0].reset_index(drop=True)
    print(f"\nRows to score (step > 0): {len(target)}")

    already_done_keys: set = set()
    existing_out: pd.DataFrame | None = None
    if args.output.exists():
        existing_out = pd.read_csv(args.output)
        for _, r in existing_out.iterrows():
            already_done_keys.add((
                r["qid"], r["group"], r["instruction_type"],
                int(r["run"]), int(r["step"]),
            ))
        print(f"  Resume: {len(already_done_keys)} rows already in {args.output} — will skip.")

    candidates: list = []
    refs_baseline: list = []
    refs_consecutive: list = []
    has_consecutive: list = []
    target_rows: list = []
    skipped = 0

    print("Pairing candidates with references...")
    for _, row in tqdm(target.iterrows(), total=len(target), desc="pairing"):
        qid, group, itype, run, step = (
            row["qid"], row["group"], row["instruction_type"],
            int(row["run"]), int(row["step"]),
        )
        key = (qid, group, itype, run, step)
        if key in already_done_keys:
            continue

        cand = step_index.get(key)
        ref_b = step_index.get((qid, group, itype, run, 0))
        ref_c = step_index.get((qid, group, itype, run, step - 1))

        if cand is None or ref_b is None:
            skipped += 1
            continue

        target_rows.append(row)
        candidates.append(cand)
        refs_baseline.append(ref_b)
        if ref_c is not None:
            refs_consecutive.append(ref_c)
            has_consecutive.append(True)
        else:
            refs_consecutive.append("")
            has_consecutive.append(False)

    print(f"\nNew pairs to score: {len(candidates)}")
    if skipped:
        print(f"Skipped (missing E_0): {skipped}")

    if not candidates:
        print("Nothing to do.")
        return

    P_b, R_b, F1_b = compute_bertscore_pairs(candidates, refs_baseline, args.batch_size, device, "Baseline")

    cand_c = [candidates[i] for i, h in enumerate(has_consecutive) if h]
    ref_c = [refs_consecutive[i] for i, h in enumerate(has_consecutive) if h]
    P_c, R_c, F1_c = compute_bertscore_pairs(cand_c, ref_c, args.batch_size, device, "Consecutive")

    bp_c = [None] * len(candidates)
    br_c = [None] * len(candidates)
    bf_c = [None] * len(candidates)
    j = 0
    for i, h in enumerate(has_consecutive):
        if h:
            bp_c[i], br_c[i], bf_c[i] = P_c[j], R_c[j], F1_c[j]
            j += 1

    new_df = pd.DataFrame({
        "qid": [r["qid"] for r in target_rows],
        "group": [r["group"] for r in target_rows],
        "instruction_type": [r["instruction_type"] for r in target_rows],
        "run": [int(r["run"]) for r in target_rows],
        "step": [int(r["step"]) for r in target_rows],
        "bert_precision_baseline": P_b,
        "bert_recall_baseline": R_b,
        "bert_f1_baseline": F1_b,
        "bert_precision_consecutive": bp_c,
        "bert_recall_consecutive": br_c,
        "bert_f1_consecutive": bf_c,
    })

    if existing_out is not None and len(existing_out) > 0:
        out = pd.concat([existing_out, new_df], ignore_index=True)
    else:
        out = new_df

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}  ({len(out)} total rows)")

    print("\n" + "=" * 70)
    print("BERTScore F1 — mean per step (all rows in output)")
    print("=" * 70)
    print("\nBaseline (vs E_0):")
    print(out.groupby("step")["bert_f1_baseline"].mean().round(4))
    print("\nConsecutive (vs E_{t-1}):")
    print(out.groupby("step")["bert_f1_consecutive"].mean().round(4))


if __name__ == "__main__":
    main()
