"""Perplexity evaluation under Llama-3.1-70B (4-bit NF4) — dataset-agnostic.

Sliding-window PPL of each E_t, same logic as scripts/_common/perplexity_eval.py
but pinned to the Llama-70B rewriter so PPL reflects "naturalness under the
rewriter itself" (consistent with the analysis convention).

Use --input / --output to point at any chain CSV.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "llama70b"))
from _common.llama_constants import CHAIN_KEYS, LLAMA_MODEL_ID  # noqa: E402
from _common.llama_model import hf_login_if_token, load_llama  # noqa: E402


@torch.no_grad()
def compute_perplexity(text, tokenizer, model, max_length=2048, stride=1024):
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"]
    seq_len = input_ids.size(1)
    if seq_len < 2:
        return float("nan")
    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end
        input_chunk = input_ids[:, begin:end].to(model.device)
        labels = input_chunk.clone()
        labels[:, :-target_len] = -100
        out = model(input_chunk, labels=labels)
        nlls.append(out.loss.item() * target_len)
        prev_end = end
        if end == seq_len:
            break
    return math.exp(sum(nlls) / seq_len)


def main():
    ap = argparse.ArgumentParser(description="Perplexity (Llama-3.1-70B) on rewriting chains.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--model", default=LLAMA_MODEL_ID)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--use-4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="use_4bit", action="store_false")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    out_path = args.output or args.input.with_name(args.input.stem + "_perplexity.csv")
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    hf_login_if_token()
    df = pd.read_csv(args.input)
    missing = [c for c in CHAIN_KEYS + ["step", "text"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    if args.smoke_test:
        first_qid = df["qid"].iloc[0]
        df = df[(df["qid"] == first_qid) & (df["run"] == 0)]
        print(f"*** SMOKE TEST: qid={first_qid}, {len(df)} rows ***")

    e0 = df[df["step"] == 0].drop_duplicates(subset=["qid", "run"], keep="first")
    rest = df[df["step"] > 0]
    to_eval = pd.concat([e0, rest], ignore_index=True)
    to_eval = to_eval.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    if out_path.exists() and not args.smoke_test:
        done = pd.read_csv(out_path)
        done_keys = set(zip(done["qid"], done["group"], done["instruction_type"],
                            done["run"], done["step"]))
        to_eval = to_eval[~to_eval.apply(
            lambda r: (r["qid"], r["group"], r["instruction_type"],
                       int(r["run"]), int(r["step"])) in done_keys,
            axis=1,
        )].reset_index(drop=True)
        print(f"  resume: {len(done_keys)} done, {len(to_eval)} remaining")

    if to_eval.empty:
        print("Nothing to compute."); return

    tokenizer, model = load_llama(args.model, use_4bit=args.use_4bit, padding_side="right")

    results = []; t_start = time.time(); total = len(to_eval)
    for _, row in to_eval.iterrows():
        text = str(row["text"]).strip()
        if not text or text.lower() == "nan":
            ppl = float("nan")
        else:
            ppl = compute_perplexity(text, tokenizer, model, args.max_length, args.stride)
        results.append({
            **{k: row[k] for k in CHAIN_KEYS},
            "step": int(row["step"]),
            "n_tokens": int(row["n_tokens"]) if pd.notna(row.get("n_tokens")) else None,
            "perplexity": round(ppl, 4),
        })
        n_done = len(results); elapsed = time.time() - t_start
        eta = (total - n_done) * elapsed / max(n_done, 1)
        label = f"{row['group']}/{row['instruction_type']}/run{row['run']}/step{row['step']}"
        print(f"[{n_done}/{total}] {label}  PPL={ppl:.2f}  ETA {eta/60:.1f} min", flush=True)

    results_df = pd.DataFrame(results)

    # Broadcast E_0
    step0 = results_df[results_df["step"] == 0]
    step_gt0 = results_df[results_df["step"] > 0]
    if not step0.empty:
        all_chains = df[CHAIN_KEYS].drop_duplicates()
        step0_broadcast = all_chains.merge(
            step0.drop(columns=["group", "instruction_type"]),
            on=["qid", "run"], how="inner",
        )
        results_df = pd.concat([step0_broadcast, step_gt0], ignore_index=True)
        results_df = results_df.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.smoke_test:
        prev = pd.read_csv(out_path)
        merged = pd.concat([prev, results_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=CHAIN_KEYS + ["step"], keep="last")
        merged.sort_values(CHAIN_KEYS + ["step"], inplace=True)
        merged.to_csv(out_path, index=False)
    else:
        results_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    pivot = results_df.pivot_table(
        index="instruction_type", columns="step", values="perplexity", aggfunc="mean",
    )
    print(pivot.round(2))


if __name__ == "__main__":
    main()
