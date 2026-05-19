"""
Perplexity evaluation for the 300q rewriting chains.

Measures text fluency/naturalness using a model different from the rewriter
to avoid self-referential bias. OLMo-3.1-7B-Instruct is used by default.

For each (qid, group, instruction_type, run, step):
  - compute perplexity of E_t under the eval model
  - step 0 is included as the baseline ceiling

Perplexity is computed as exp(mean NLL over tokens), using sliding-window
approach for long texts (stride = context_len // 2) to handle texts longer
than the model's context window.

Output CSV (one row per chain step, including step 0):
  qid, group, instruction_type, run, step, n_tokens, perplexity

Uso:
  python3 scripts/300q/perplexity_eval_300q.py
  python3 scripts/300q/perplexity_eval_300q.py --model allenai/OLMo-3.1-7B-Instruct
  python3 scripts/300q/perplexity_eval_300q.py --smoke-test
  python3 scripts/300q/perplexity_eval_300q.py --save-stats
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
RESULTS_DIR = REPO_ROOT / "results" / "300q"

DEFAULT_CHAINS_CSV = RESULTS_DIR / "rewriting_chains_300q.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "rewriting_chains_300q_perplexity.csv"

DEFAULT_MODEL_ID = "allenai/OLMo-3.1-7B-Instruct"
CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]
SMOKE_QID = "2hop__635544_110949"


def load_model(model_id: str, use_4bit: bool):
    print(f"Loading {model_id} (4-bit={use_4bit})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"device_map": "auto"}
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
    print(f"  device map: {getattr(model, 'hf_device_map', 'n/a')}")
    return tokenizer, model


@torch.no_grad()
def compute_perplexity(text: str, tokenizer, model, max_length: int = 2048, stride: int = 1024) -> float:
    """Sliding-window perplexity for texts longer than max_length tokens."""
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"]
    seq_len = input_ids.size(1)

    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        # tokens we actually score in this window (not the prefix context)
        target_len = end - prev_end
        input_chunk = input_ids[:, begin:end].to(model.device)

        # labels: mask the context prefix with -100 so it doesn't contribute to loss
        labels = input_chunk.clone()
        labels[:, :-target_len] = -100

        outputs = model(input_chunk, labels=labels)
        # outputs.loss is mean NLL over non-masked tokens
        nlls.append(outputs.loss.item() * target_len)

        prev_end = end
        if end == seq_len:
            break

    total_tokens = seq_len - 0  # approximate; fine for comparison purposes
    mean_nll = sum(nlls) / total_tokens
    return math.exp(mean_nll)


def main() -> None:
    parser = argparse.ArgumentParser(description="Perplexity evaluation on 300q rewriting chains.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CHAINS_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID,
                        help=f"HF model id for perplexity (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Context window size for sliding-window PPL (default: 2048)")
    parser.add_argument("--stride", type=int, default=1024,
                        help="Stride for sliding-window PPL (default: 1024)")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Enable 4-bit NF4 quantization.")
    parser.add_argument("--smoke-test", action="store_true",
                        help=f"Run only on {SMOKE_QID}, run 0.")
    parser.add_argument("--save-stats", action="store_true",
                        help="Save breakdown CSVs to results/300q/stats/")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Chains CSV not found: {args.input}")

    print(f"Loading chains: {args.input}")
    df = pd.read_csv(args.input)

    if args.smoke_test:
        df = df[(df["qid"] == SMOKE_QID) & (df["run"] == 0)]
        print(f"*** SMOKE TEST: {len(df)} rows ***")

    # E_0 is identical across instruction_types of the same (qid, run) —
    # evaluate once per (qid, run, step=0), broadcast later.
    e0 = df[df["step"] == 0].drop_duplicates(subset=["qid", "run"], keep="first")
    rest = df[df["step"] > 0]
    to_eval = pd.concat([e0, rest], ignore_index=True)
    to_eval = to_eval.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    # Resume support
    if args.output.exists() and not args.smoke_test:
        done = pd.read_csv(args.output)
        done_keys = set(zip(done["qid"], done["group"], done["instruction_type"], done["run"], done["step"]))
        to_eval = to_eval[~to_eval.apply(
            lambda r: (r["qid"], r["group"], r["instruction_type"], int(r["run"]), int(r["step"])) in done_keys,
            axis=1
        )].reset_index(drop=True)
        print(f"  resuming — {len(done_keys)} already done, {len(to_eval)} remaining")

    if to_eval.empty:
        print("Nothing to compute.")
        return

    tokenizer, model = load_model(args.model, args.use_4bit)

    results = []
    t_start = time.time()
    total = len(to_eval)

    for i, row in to_eval.iterrows():
        text = str(row["text"]).strip()
        if not text:
            ppl = float("nan")
        else:
            ppl = compute_perplexity(text, tokenizer, model, args.max_length, args.stride)

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
        print(f"[{n_done}/{total}] {label}  PPL={ppl:.2f}  ETA {eta/60:.1f} min")

    results_df = pd.DataFrame(results)

    # Broadcast E_0 perplexity to all instruction_types of the same (qid, run)
    step0 = results_df[results_df["step"] == 0]
    step_gt0 = results_df[results_df["step"] > 0]
    if not step0.empty:
        all_chains = df[CHAIN_KEYS].drop_duplicates()
        if args.smoke_test:
            all_chains = all_chains[(all_chains["qid"] == SMOKE_QID) & (all_chains["run"] == 0)]
        step0_broadcast = all_chains.merge(
            step0.drop(columns=["group", "instruction_type"]),
            on=["qid", "run"], how="inner",
        )
        results_df = pd.concat([step0_broadcast, step_gt0], ignore_index=True)
        results_df = results_df.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.smoke_test:
        prev = pd.read_csv(args.output)
        merged = pd.concat([prev, results_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=CHAIN_KEYS + ["step"], keep="last")
        merged.sort_values(CHAIN_KEYS + ["step"], inplace=True)
        merged.to_csv(args.output, index=False)
    else:
        results_df.to_csv(args.output, index=False)

    print(f"\nSaved: {args.output}")
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
        stats_dir = RESULTS_DIR / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        pivot.round(4).to_csv(stats_dir / "perplexity_by_instruction_step.csv")
        results_df.groupby(["step"])["perplexity"].agg(["mean","std","count"]).round(4).to_csv(
            stats_dir / "perplexity_by_step.csv"
        )
        results_df.groupby(["instruction_type"])["perplexity"].agg(["mean","std","count"]).round(4).to_csv(
            stats_dir / "perplexity_by_instruction.csv"
        )
        print(f"[saved] {stats_dir}/perplexity_*.csv")


if __name__ == "__main__":
    main()
