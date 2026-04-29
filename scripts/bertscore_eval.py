"""
BERTScore evaluation (standalone) for the 15q pilot.

Computes BERTScore in both modes defined in §6.2:
  - Baseline:    sim(E_t, E_0)   — cumulative drift from the original
  - Consecutive: sim(E_t, E_{t-1}) — change between adjacent steps

Standalone: reads the chain CSV directly, no dependency on a precomputed
FactScore file. Each row at step > 0 gets paired with E_0 and E_{t-1} of the
same chain.

Output CSV schema (one row per source row, step > 0 only):
  qid, group, instruction_type, run, step,
  bert_precision_baseline, bert_recall_baseline, bert_f1_baseline,
  bert_precision_consecutive, bert_recall_consecutive, bert_f1_consecutive
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from bert_score import score as compute_bert_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "results" / "rewriting_chains_15q.csv"
DEFAULT_OUT = REPO_ROOT / "results" / "rewriting_chains_15q_bertscore.csv"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

BERT_MODEL = "roberta-large"
BERT_NUM_LAYERS = 17
BERT_LANG = "en"


def build_step_index(chains_df: pd.DataFrame) -> dict:
    """Return {(qid, group, instr_type, run, step): text} for every row."""
    idx = {}
    for _, row in chains_df.iterrows():
        key = (
            row["qid"], row["group"], row["instruction_type"],
            int(row["run"]), int(row["step"]),
        )
        idx[key] = row["text"]
    return idx


def compute_bertscore_pairs(candidates, references, batch_size, device, label):
    """Run BERTScore on aligned (candidate, reference) lists with progress bar."""
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
        verbose=True,  # bert-score's own progress bar
    )
    dt = time.time() - t0
    print(f"  [{label}] done in {dt:.1f}s ({len(candidates)/dt:.1f} pairs/s)")
    return P.tolist(), R.tolist(), F1.tolist()


def main():
    parser = argparse.ArgumentParser(description="Compute BERTScore (Baseline + Consecutive) on a chain CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV, help=f"Input chain CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help=f"Output BERTScore CSV (default: {DEFAULT_OUT})")
    parser.add_argument("--batch-size", type=int, default=8, help="BERTScore batch size (default: 8 — keep small on CPU/Mac).")
    parser.add_argument("--device", default=None, help="Force device (cuda / cpu / mps). Default: auto-detect.")
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
    print(f"BERTScore (Baseline + Consecutive)  —  device={device}, batch={args.batch_size}")
    print("=" * 70)

    if not args.input.exists():
        raise FileNotFoundError(f"Chain CSV not found: {args.input}")

    print(f"\nLoading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  rows: {len(df)}  |  unique chains: {df.groupby(CHAIN_KEYS).ngroups}")

    print("Building step index...")
    step_index = build_step_index(df)
    print(f"  {len(step_index)} (chain, step) entries")

    # Build the list of pairs to score.
    # We score every step > 0. For each, Baseline pairs (E_t, E_0) and Consecutive pairs (E_t, E_{t-1}).
    target = df[df["step"] > 0].reset_index(drop=True)
    print(f"\nRows to score (step > 0): {len(target)}")

    candidates: list = []
    refs_baseline: list = []
    refs_consecutive: list = []
    has_consecutive: list = []
    skipped = 0

    print("Pairing candidates with references...")
    for _, row in tqdm(target.iterrows(), total=len(target), desc="pairing"):
        qid, group, itype, run, step = (
            row["qid"], row["group"], row["instruction_type"],
            int(row["run"]), int(row["step"]),
        )
        cand = step_index.get((qid, group, itype, run, step))
        ref_b = step_index.get((qid, group, itype, run, 0))
        ref_c = step_index.get((qid, group, itype, run, step - 1))

        if cand is None or ref_b is None:
            candidates.append("")
            refs_baseline.append("")
            refs_consecutive.append("")
            has_consecutive.append(False)
            skipped += 1
            continue

        candidates.append(cand)
        refs_baseline.append(ref_b)
        if ref_c is not None:
            refs_consecutive.append(ref_c)
            has_consecutive.append(True)
        else:
            refs_consecutive.append("")
            has_consecutive.append(False)

    valid_b = [i for i, c in enumerate(candidates) if c and refs_baseline[i]]
    valid_c = [i for i, _ in enumerate(candidates) if has_consecutive[i] and candidates[i]]

    print(f"\nBaseline pairs:    {len(valid_b)}")
    print(f"Consecutive pairs: {len(valid_c)}")
    if skipped:
        print(f"Skipped (missing E_0 or E_t): {skipped}")
    print(f"BERTScore model: {BERT_MODEL} (layer {BERT_NUM_LAYERS})")
    print()

    # Baseline mode: sim(E_t, E_0)
    cand_b = [candidates[i] for i in valid_b]
    ref_b = [refs_baseline[i] for i in valid_b]
    P_b, R_b, F1_b = compute_bertscore_pairs(cand_b, ref_b, args.batch_size, device, "Baseline")

    # Consecutive mode: sim(E_t, E_{t-1})
    cand_c = [candidates[i] for i in valid_c]
    ref_c = [refs_consecutive[i] for i in valid_c]
    P_c, R_c, F1_c = compute_bertscore_pairs(cand_c, ref_c, args.batch_size, device, "Consecutive")

    bp_b = [None] * len(target)
    br_b = [None] * len(target)
    bf_b = [None] * len(target)
    bp_c = [None] * len(target)
    br_c = [None] * len(target)
    bf_c = [None] * len(target)

    for k, i in enumerate(valid_b):
        bp_b[i], br_b[i], bf_b[i] = P_b[k], R_b[k], F1_b[k]
    for k, i in enumerate(valid_c):
        bp_c[i], br_c[i], bf_c[i] = P_c[k], R_c[k], F1_c[k]

    out = target[CHAIN_KEYS + ["step"]].copy()
    out["bert_precision_baseline"] = bp_b
    out["bert_recall_baseline"] = br_b
    out["bert_f1_baseline"] = bf_b
    out["bert_precision_consecutive"] = bp_c
    out["bert_recall_consecutive"] = br_c
    out["bert_f1_consecutive"] = bf_c

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")

    print("\n" + "=" * 70)
    print("BERTScore F1 — median per (instruction_type, step)")
    print("=" * 70)
    print("\nBaseline (vs E_0):")
    pivot_b = out.pivot_table(
        index="instruction_type", columns="step", values="bert_f1_baseline", aggfunc="median",
    )
    print(pivot_b.round(3))

    print("\nConsecutive (vs E_{t-1}):")
    pivot_c = out.pivot_table(
        index="instruction_type", columns="step", values="bert_f1_consecutive", aggfunc="median",
    )
    print(pivot_c.round(3))


if __name__ == "__main__":
    main()
