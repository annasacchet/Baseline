"""
BERTScore evaluation — OPTIMIZED GPU version.

Same pipeline, model (roberta-large layer 17), pair construction, and
output schema as scripts/15q/bertscore_eval.py. The ONLY differences are
GPU-side:

  * `bert-score` already batches internally; the default --batch-size is
    raised from 8 → 128 (lower it via --batch-size if you OOM).
  * --nthreads exposed (forwarded to bert-score) to reduce CPU stalls on
    tokenization.
  * idf=False is set explicitly (the default), to make it clear no
    IDF-corpus pass is needed.
  * On CUDA, autocast bfloat16 wraps the scoring calls.

Outputs are unchanged: same CSV layout, same columns, same path.
"""

import argparse
import time
from pathlib import Path
from contextlib import nullcontext

import pandas as pd
import torch
from bert_score import score as compute_bert_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q.csv"
DEFAULT_OUT = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q_bertscore.csv"

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


def compute_bertscore_pairs(
    candidates, references, batch_size, device, label, nthreads, use_amp,
):
    if not candidates:
        return [], [], []
    print(f"  [{label}] computing BERTScore on {len(candidates)} pairs...")
    t0 = time.time()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_amp and device == "cuda")
        else nullcontext()
    )
    with amp_ctx:
        P, R, F1 = compute_bert_score(
            candidates,
            references,
            lang=BERT_LANG,
            model_type=BERT_MODEL,
            num_layers=BERT_NUM_LAYERS,
            batch_size=batch_size,
            device=device,
            nthreads=nthreads,
            idf=False,
            verbose=True,
        )
    dt = time.time() - t0
    print(f"  [{label}] done in {dt:.1f}s ({len(candidates)/dt:.1f} pairs/s)")
    return P.tolist(), R.tolist(), F1.tolist()


def main():
    parser = argparse.ArgumentParser(description="BERTScore (Baseline + Consecutive) — optimized.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="BERTScore batch size (default: 128 on GPU). Lower if you OOM.",
    )
    parser.add_argument("--device", default=None, help="Force device (cuda/cpu/mps). Default: auto-detect.")
    parser.add_argument("--nthreads", type=int, default=4,
                        help="CPU threads for bert-score tokenization (default: 4).")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable bfloat16 autocast on CUDA (default: on for cuda).")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    use_amp = not args.no_amp

    print("=" * 70)
    print(f"BERTScore (OPTIMIZED)  —  device={device}, batch={args.batch_size}, "
          f"nthreads={args.nthreads}, amp={use_amp and device == 'cuda'}")
    print("=" * 70)

    if not args.input.exists():
        raise FileNotFoundError(f"Chain CSV not found: {args.input}")

    print(f"\nLoading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  rows: {len(df)}  |  unique chains: {df.groupby(CHAIN_KEYS).ngroups}")

    print("Building step index...")
    step_index = build_step_index(df)
    print(f"  {len(step_index)} (chain, step) entries")

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

    cand_b = [candidates[i] for i in valid_b]
    ref_b = [refs_baseline[i] for i in valid_b]
    P_b, R_b, F1_b = compute_bertscore_pairs(
        cand_b, ref_b, args.batch_size, device, "Baseline", args.nthreads, use_amp,
    )

    cand_c = [candidates[i] for i in valid_c]
    ref_c = [refs_consecutive[i] for i in valid_c]
    P_c, R_c, F1_c = compute_bertscore_pairs(
        cand_c, ref_c, args.batch_size, device, "Consecutive", args.nthreads, use_amp,
    )

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
