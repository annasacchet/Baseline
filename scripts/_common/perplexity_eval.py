"""
Perplexity evaluation for the rewriting chains — MuSiQue 300q and NewsQA 100q.

Measures text fluency/naturalness of each rewritten E_t. Perplexity is the
exponentiated mean token NLL under the eval model; a sliding window
(stride = max_length // 2) handles texts longer than the context window.

The eval model is OLMo-3.1-32B-Instruct — the same model used as the rewriter.
This means the metric reflects how natural the text is *to the rewriter itself*;
acceptable here because both datasets are scored with the identical model, so
PPL is comparable across steps and across datasets, which is what the analysis
needs. (Use --model to swap in a held-out model if a bias-free PPL is wanted.)

For each (qid, group, instruction_type, run, step):
  - compute perplexity of E_t under the eval model
  - step 0 (E_0) is included as the baseline ceiling
  - E_0 is identical across instruction_types of the same (qid, run): it is
    scored once and broadcast, to avoid redundant forward passes.

Output CSV (one row per chain step, including step 0):
  qid, group, instruction_type, run, step, n_tokens, perplexity

Usage:
  # MuSiQue 300q
  python3 scripts/_common/perplexity_eval.py --dataset 300q --use-4bit
  # NewsQA 100q
  python3 scripts/_common/perplexity_eval.py --dataset newsqa --use-4bit
  # quick sanity check
  python3 scripts/_common/perplexity_eval.py --dataset 300q --smoke-test
  # explicit paths / model override
  python3 scripts/_common/perplexity_eval.py --input ... --output ... --model ...
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_MODEL_ID = "allenai/OLMo-3.1-32B-Instruct"
CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

# Per-dataset input/output paths and the qid used for the smoke test.
DATASETS = {
    "300q": {
        "input": RESULTS_DIR / "300q" / "rewriting_chains_300q.csv",
        "output": RESULTS_DIR / "300q" / "rewriting_chains_300q_perplexity.csv",
        "smoke_qid": "2hop__635544_110949",
    },
    "newsqa": {
        "input": RESULTS_DIR / "newsqa" / "rewriting_chains_newsqa_100q.csv",
        "output": RESULTS_DIR / "newsqa" / "rewriting_chains_newsqa_100q_perplexity.csv",
        "smoke_qid": None,  # set to the first qid in the CSV at runtime
    },
}


def load_model(model_id: str, use_4bit: bool):
    print(f"Loading {model_id} (4-bit={use_4bit})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"device_map": "auto", "trust_remote_code": True}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    print(f"  device map: {getattr(model, 'hf_device_map', 'n/a')}", flush=True)
    return tokenizer, model


@torch.no_grad()
def compute_perplexity(text: str, tokenizer, model, max_length: int = 2048,
                       stride: int = 1024) -> float:
    """Sliding-window perplexity for texts longer than max_length tokens."""
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"]
    seq_len = input_ids.size(1)
    if seq_len < 2:
        return float("nan")  # need at least one token to predict

    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        # tokens we actually score in this window (not the prefix context)
        target_len = end - prev_end
        input_chunk = input_ids[:, begin:end].to(model.device)

        # labels: mask the context prefix with -100 so it doesn't add to loss
        labels = input_chunk.clone()
        labels[:, :-target_len] = -100

        outputs = model(input_chunk, labels=labels)
        # outputs.loss is mean NLL over non-masked tokens
        nlls.append(outputs.loss.item() * target_len)

        prev_end = end
        if end == seq_len:
            break

    mean_nll = sum(nlls) / seq_len
    return math.exp(mean_nll)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perplexity evaluation on rewriting chains (300q / newsqa)."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), default=None,
                        help="Pick dataset to set default --input/--output. "
                             "Omit only if both --input and --output are given.")
    parser.add_argument("--input", type=Path, default=None,
                        help="Chains CSV (overrides --dataset default).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV (overrides --dataset default).")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID,
                        help=f"HF model id for perplexity (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Context window size for sliding-window PPL (default: 2048)")
    parser.add_argument("--stride", type=int, default=1024,
                        help="Stride for sliding-window PPL (default: 1024)")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Enable 4-bit NF4 quantization (recommended for the 32B model).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run only on one qid, run 0.")
    parser.add_argument("--save-stats", action="store_true",
                        help="Save breakdown CSVs next to the output file.")
    args = parser.parse_args()

    # Resolve input/output from --dataset unless given explicitly.
    if args.dataset:
        cfg = DATASETS[args.dataset]
        in_path = args.input or cfg["input"]
        out_path = args.output or cfg["output"]
        smoke_qid = cfg["smoke_qid"]
    else:
        if not (args.input and args.output):
            parser.error("provide --dataset, or both --input and --output")
        in_path, out_path, smoke_qid = args.input, args.output, None

    if not in_path.exists():
        raise FileNotFoundError(f"Chains CSV not found: {in_path}")

    print(f"Dataset : {args.dataset or 'custom'}")
    print(f"Input   : {in_path}")
    print(f"Output  : {out_path}")
    print(f"Loading chains: {in_path}", flush=True)
    df = pd.read_csv(in_path)

    missing_cols = [c for c in CHAIN_KEYS + ["step", "text"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV missing required columns: {missing_cols}")

    if args.smoke_test:
        if smoke_qid is None or smoke_qid not in set(df["qid"]):
            smoke_qid = df["qid"].iloc[0]
        df = df[(df["qid"] == smoke_qid) & (df["run"] == 0)]
        print(f"*** SMOKE TEST: qid={smoke_qid}, {len(df)} rows ***")

    # E_0 is identical across instruction_types of the same (qid, run) —
    # evaluate once per (qid, run, step=0), broadcast later.
    e0 = df[df["step"] == 0].drop_duplicates(subset=["qid", "run"], keep="first")
    rest = df[df["step"] > 0]
    to_eval = pd.concat([e0, rest], ignore_index=True)
    to_eval = to_eval.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    # Resume support
    if out_path.exists() and not args.smoke_test:
        done = pd.read_csv(out_path)
        done_keys = set(zip(done["qid"], done["group"], done["instruction_type"],
                            done["run"], done["step"]))
        to_eval = to_eval[~to_eval.apply(
            lambda r: (r["qid"], r["group"], r["instruction_type"],
                       int(r["run"]), int(r["step"])) in done_keys,
            axis=1,
        )].reset_index(drop=True)
        print(f"  resuming — {len(done_keys)} already done, {len(to_eval)} remaining")

    if to_eval.empty:
        print("Nothing to compute.")
        return

    tokenizer, model = load_model(args.model, args.use_4bit)

    results = []
    t_start = time.time()
    total = len(to_eval)

    for _, row in to_eval.iterrows():
        text = str(row["text"]).strip()
        if not text or text.lower() == "nan":
            ppl = float("nan")
        else:
            ppl = compute_perplexity(text, tokenizer, model,
                                     args.max_length, args.stride)

        results.append({
            **{k: row[k] for k in CHAIN_KEYS},
            "step": int(row["step"]),
            "n_tokens": int(row["n_tokens"]) if pd.notna(row.get("n_tokens")) else None,
            "perplexity": round(ppl, 4),
        })

        n_done = len(results)
        elapsed = time.time() - t_start
        eta = (total - n_done) * elapsed / max(n_done, 1)
        label = f"{row['group']}/{row['instruction_type']}/run{row['run']}/step{row['step']}"
        print(f"[{n_done}/{total}] {label}  PPL={ppl:.2f}  ETA {eta/60:.1f} min", flush=True)

    results_df = pd.DataFrame(results)

    # Broadcast E_0 perplexity to all instruction_types of the same (qid, run)
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
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")

    # Summary
    print("\n" + "=" * 60)
    print("Perplexity — mean per (instruction_type, step)")
    print("=" * 60)
    pivot = results_df.pivot_table(
        index="instruction_type", columns="step", values="perplexity", aggfunc="mean"
    )
    print(pivot.round(2))

    if args.save_stats:
        stats_dir = out_path.parent / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        stem = out_path.stem
        pivot.round(4).to_csv(stats_dir / f"{stem}_by_instruction_step.csv")
        results_df.groupby(["step"])["perplexity"].agg(["mean", "std", "count"]).round(4).to_csv(
            stats_dir / f"{stem}_by_step.csv"
        )
        results_df.groupby(["instruction_type"])["perplexity"].agg(["mean", "std", "count"]).round(4).to_csv(
            stats_dir / f"{stem}_by_instruction.csv"
        )
        print(f"[saved] {stats_dir}/{stem}_*.csv")


if __name__ == "__main__":
    main()
