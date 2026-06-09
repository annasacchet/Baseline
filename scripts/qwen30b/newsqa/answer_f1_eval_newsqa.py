"""Answer F1 on NewsQA chains — QA model = OLMo-3.1-32B via vLLM.

Extractive prompt (NewsQA answers are verbatim spans), aliases pulled from
the chain CSV (||-joined). Max over (gold + aliases) like the official metric.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "qwen30b"))
from _common.f1_utils import best_f1  # noqa: E402
from _common.olmo_constants import ALIAS_SEP, CHAIN_KEYS, OLMO_MODEL_ID  # noqa: E402
from _common.olmo_vllm import (  # noqa: E402
    generate_batch_vllm,
    hf_login_if_token,
    load_vllm,
    render_chat,
)

DEFAULT_CHAINS_CSV = REPO_ROOT / "results" / "qwen30b" / "newsqa" / "rewriting_chains_newsqa.csv"

QA_USER_TEMPLATE = """Answer the question using only words copied verbatim from the context below. \
Your answer must be a continuous span of text that appears exactly in the context — do not paraphrase, \
do not add words not in the context.

Context:
{context}

Question: {question}
Answer (verbatim span from context):"""


def parse_aliases(value, fallback_gold):
    aliases = []
    if isinstance(value, str) and value.strip():
        aliases = [a.strip() for a in value.split(ALIAS_SEP) if a and a.strip()]
    if fallback_gold and fallback_gold not in aliases:
        aliases.insert(0, fallback_gold)
    return aliases


def build_prompts(tokenizer, rows):
    prompts = []
    for row in rows:
        user = QA_USER_TEMPLATE.format(
            context=str(row["text"]).strip(),
            question=str(row["question"]).strip(),
        )
        prompts.append(render_chat(tokenizer, None, user))
    return prompts


def main():
    ap = argparse.ArgumentParser(description="Answer F1 on NewsQA chains, OLMo-3.1-32B QA via vLLM.")
    ap.add_argument("--input", type=Path, default=DEFAULT_CHAINS_CSV)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--model", default=OLMO_MODEL_ID)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    chains_csv = args.input
    output_csv = args.output or chains_csv.with_name(chains_csv.stem + "_answer_f1_span.csv")
    if not chains_csv.exists():
        raise FileNotFoundError(chains_csv)

    hf_login_if_token()
    df = pd.read_csv(chains_csv)
    if "gold_answer_aliases" not in df.columns:
        df["gold_answer_aliases"] = ""
    df = df.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    to_eval = df.copy()
    if args.smoke_test:
        first_qid = df["qid"].iloc[0]
        to_eval = to_eval[(to_eval["qid"] == first_qid) & (to_eval["run"] == 0)]
        print(f"*** SMOKE TEST: {len(to_eval)} rows ***")

    e0 = to_eval[to_eval["step"] == 0].drop_duplicates(subset=["qid", "run"], keep="first")
    to_eval = pd.concat([e0, to_eval[to_eval["step"] > 0]], ignore_index=True)
    to_eval = to_eval.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    if to_eval.empty:
        raise RuntimeError("No rows to evaluate.")

    total = len(to_eval)
    print(f"Answer F1 on {total} rows — QA = {args.model}")
    llm, tokenizer = load_vllm(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    rows = to_eval.to_dict(orient="records")
    print(f"Building {total} prompts ...")
    prompts = build_prompts(tokenizer, rows)

    print("Generating ...")
    t0 = time.time()
    preds = generate_batch_vllm(llm, prompts, temperature=0.0, max_new_tokens=args.max_new_tokens)
    print(f"  vLLM generated {total} answers in {time.time() - t0:.1f}s")

    results = []
    for row, pred in zip(rows, preds):
        gold = str(row["gold_answer"])
        aliases = parse_aliases(row.get("gold_answer_aliases"), gold)
        f1, matched_ref = best_f1(pred, aliases)
        results.append({
            **{k: row[k] for k in CHAIN_KEYS},
            "step": int(row["step"]),
            "question": row["question"], "gold_answer": gold,
            "gold_answer_aliases": ALIAS_SEP.join(aliases),
            "predicted_answer": pred, "matched_reference": matched_ref,
            "answer_f1": f1,
        })

    results_df = pd.DataFrame(results)

    # Broadcast E_0
    if not results_df.empty:
        step0 = results_df[results_df["step"] == 0]
        step_gt0 = results_df[results_df["step"] > 0]
        if not step0.empty:
            all_chains = df[CHAIN_KEYS].drop_duplicates()
            if args.smoke_test:
                first_qid = df["qid"].iloc[0]
                all_chains = all_chains[(all_chains["qid"] == first_qid) & (all_chains["run"] == 0)]
            step0_broadcast = all_chains.merge(
                step0.drop(columns=["group", "instruction_type"]),
                on=["qid", "run"], how="inner",
            )
            results_df = pd.concat([step0_broadcast, step_gt0], ignore_index=True)
            results_df = results_df.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        prev = pd.read_csv(output_csv)
        merged = pd.concat([prev, results_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=CHAIN_KEYS + ["step"], keep="last")
        merged.to_csv(output_csv, index=False)
    else:
        results_df.to_csv(output_csv, index=False)

    print(f"\nSaved: {output_csv}")
    pivot = results_df.pivot_table(
        index=["group", "instruction_type"], columns="step", values="answer_f1", aggfunc="mean",
    )
    print(pivot.round(3))


if __name__ == "__main__":
    main()
