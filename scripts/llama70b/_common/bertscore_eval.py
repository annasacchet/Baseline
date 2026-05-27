"""BERTScore (Baseline + Consecutive) — dataset-agnostic.

Mirrors scripts/newsqa/bertscore_eval_newsqa.py. Use --input / --output to point
at any of the Llama-70B chain CSVs (musique / newsqa / fictionalqa). The text
pairs are built directly from CHAIN_KEYS + step columns; the rewriter model
is irrelevant here because BERTScore uses roberta-large internally.

Two modes are computed:
  - Baseline:    sim(E_t, E_0)
  - Consecutive: sim(E_t, E_{t-1})
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "llama70b"))
from _common.llama_constants import CHAIN_KEYS  # noqa: E402

BERT_MODEL = "roberta-large"
BERT_NUM_LAYERS = 17
BERT_LANG = "en"


def build_step_index(df):
    return {
        (r["qid"], r["group"], r["instruction_type"], int(r["run"]), int(r["step"])): r["text"]
        for _, r in df.iterrows()
    }


def compute_bertscore_pairs(candidates, references, batch_size, device, label):
    from bert_score import score as compute_bert_score
    if not candidates:
        return [], [], []
    print(f"  [{label}] computing BERTScore on {len(candidates)} pairs ...")
    t0 = time.time()
    P, R, F1 = compute_bert_score(
        candidates, references, lang=BERT_LANG,
        model_type=BERT_MODEL, num_layers=BERT_NUM_LAYERS,
        batch_size=batch_size, device=device, verbose=True,
    )
    print(f"  [{label}] done in {time.time() - t0:.1f}s")
    return P.tolist(), R.tolist(), F1.tolist()


def main():
    ap = argparse.ArgumentParser(description="BERTScore on Llama-70B rewriting chains.")
    ap.add_argument("--input", type=Path, required=True, help="Chains CSV")
    ap.add_argument("--output", type=Path, default=None,
                    help="Default: <input_stem>_bertscore.csv next to input.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    out_path = args.output or args.input.with_name(args.input.stem + "_bertscore.csv")

    print("=" * 70)
    print(f"BERTScore  —  device={device}, batch={args.batch_size}")
    print(f"  model: {BERT_MODEL} (layer {BERT_NUM_LAYERS})")
    print(f"  input : {args.input}")
    print(f"  output: {out_path}")
    print("=" * 70)

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    df = pd.read_csv(args.input)
    print(f"  rows: {len(df)} | chains: {df.groupby(CHAIN_KEYS).ngroups}")

    step_index = build_step_index(df)
    target = df[df["step"] > 0].reset_index(drop=True)
    print(f"  rows to score (step > 0): {len(target)}")

    already_done_keys = set()
    existing_out = None
    if out_path.exists():
        existing_out = pd.read_csv(out_path)
        for _, r in existing_out.iterrows():
            already_done_keys.add((r["qid"], r["group"], r["instruction_type"],
                                   int(r["run"]), int(r["step"])))
        print(f"  Resume: {len(already_done_keys)} rows already in {out_path.name}")

    cands, refs_b, refs_c, has_c, target_rows = [], [], [], [], []
    skipped = 0
    for _, row in tqdm(target.iterrows(), total=len(target), desc="pairing"):
        key = (row["qid"], row["group"], row["instruction_type"],
               int(row["run"]), int(row["step"]))
        if key in already_done_keys:
            continue
        cand = step_index.get(key)
        ref_b = step_index.get((row["qid"], row["group"], row["instruction_type"], int(row["run"]), 0))
        ref_c = step_index.get((row["qid"], row["group"], row["instruction_type"], int(row["run"]), int(row["step"]) - 1))
        if cand is None or ref_b is None:
            skipped += 1; continue
        target_rows.append(row); cands.append(cand); refs_b.append(ref_b)
        if ref_c is not None:
            refs_c.append(ref_c); has_c.append(True)
        else:
            refs_c.append(""); has_c.append(False)

    print(f"  new pairs: {len(cands)}  skipped: {skipped}")
    if not cands:
        print("Nothing to do.")
        return

    P_b, R_b, F1_b = compute_bertscore_pairs(cands, refs_b, args.batch_size, device, "Baseline")
    cand_c = [cands[i] for i, h in enumerate(has_c) if h]
    ref_c = [refs_c[i] for i, h in enumerate(has_c) if h]
    P_c, R_c, F1_c = compute_bertscore_pairs(cand_c, ref_c, args.batch_size, device, "Consecutive")

    bp_c = [None] * len(cands); br_c = [None] * len(cands); bf_c = [None] * len(cands)
    j = 0
    for i, h in enumerate(has_c):
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

    out = pd.concat([existing_out, new_df], ignore_index=True) if (existing_out is not None and len(existing_out) > 0) else new_df
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(out)} rows)")

    print("\nBERTScore F1 — mean per step")
    print("Baseline:", out.groupby("step")["bert_f1_baseline"].mean().round(4).to_dict())
    print("Consecutive:", out.groupby("step")["bert_f1_consecutive"].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
