"""FictionalQA rewriting chain pipeline — Llama-3.1-70B-Instruct (4-bit NF4).

Same structure as scripts/fictionalqa/rewriting_pipeline_fictionalqa.py but the
rewriter is Llama-3.1-70B-Instruct in 4-bit NF4 and the prompt uses the
XML + system-prompt fix.

Reads jwkirchenbauer/fictionalqa from HF Hub. One Q per fiction document is
picked (grade_blind==0, grade_informed==1, unique). gold_answer = natural_answer.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "llama70b"))
from _common.llama_constants import (  # noqa: E402
    ALIAS_SEP,
    ALL_INSTRUCTIONS,
    CHAIN_KEYS,
    DEFAULT_SYSTEM_PROMPT,
    LLAMA_AWQ_MODEL_ID,
    REWRITE_TEMPLATE,
)
from _common.llama_vllm import (  # noqa: E402
    generate_batch_vllm,
    hf_login_if_token,
    load_vllm,
    render_chat,
)

DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "llama70b" / "fictionalqa" / "rewriting_chains_fictionalqa.csv"
HF_REPO = "jwkirchenbauer/fictionalqa"


def download_parquet(filename):
    return Path(hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=filename))


def load_fictionalqa(allowed_styles=None):
    print("  downloading FictionalQA from HF ...", flush=True)
    qa_path = download_parquet("joined_qa/train-00000-of-00001.parquet")
    fic_path = download_parquet("fictions/train-00000-of-00001.parquet")
    df_qa = pd.read_parquet(qa_path); df_fic = pd.read_parquet(fic_path)
    print(f"  joined_qa: {len(df_qa):,} rows · fictions: {len(df_fic):,} docs", flush=True)

    if allowed_styles:
        df_fic = df_fic[df_fic["style"].isin(allowed_styles)]
        df_qa = df_qa[df_qa["style"].isin(allowed_styles)]

    quality = df_qa[(df_qa["grade_blind"] == 0) & (df_qa["grade_informed"] == 1)].copy()
    quality["self_id"] = quality["question_id"]
    is_root = quality["duplicate_root"].isna() | (quality["duplicate_root"] == quality["self_id"])
    is_unique = quality["duplicate_relationship"].isin([None, "", "unique"]) | quality["duplicate_relationship"].isna()
    quality = quality[is_root | is_unique].copy()
    print(f"  quality-filtered Qs: {len(quality):,}", flush=True)

    items = []
    for fid, group in quality.groupby("fiction_id", sort=False):
        group = group.assign(_alen=group["natural_answer"].str.len())
        best = group.sort_values("_alen", na_position="last").iloc[0]
        nat = (best["natural_answer"] or "").strip()
        span = (best["span_answer"] or "").strip()
        if not nat:
            continue
        aliases = [nat] + ([span] if span and span != nat else [])
        items.append({
            "id": fid, "event_id": best["event_id"], "style": best["style"],
            "text": best["fiction"], "question": best["question"],
            "answer": nat, "aliases": aliases,
        })
    print(f"  usable fictions: {len(items):,}", flush=True)
    return items


def sample_items(items, n, seed):
    rng = random.Random(seed); pool = list(items); rng.shuffle(pool); return pool[:n]


def run_doc(llm, tokenizer, E0, specs, *, n_iterations, temperature,
            max_new_tokens, system_prompt):
    for s in specs:
        s["chain"] = [E0]; s["current"] = E0
    for _ in range(n_iterations):
        prompts = [
            render_chat(
                tokenizer, system_prompt,
                REWRITE_TEMPLATE.format(instruction=s["instruction"], text=s["current"]),
            )
            for s in specs
        ]
        outs = generate_batch_vllm(
            llm, prompts,
            temperature=temperature, max_new_tokens=max_new_tokens,
        )
        for s, new_text in zip(specs, outs):
            s["chain"].append(new_text); s["current"] = new_text
    return specs


def load_done_keys(csv_path):
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return {tuple(row[k] for k in CHAIN_KEYS) for _, row in df[CHAIN_KEYS].drop_duplicates().iterrows()}


def append_rows(csv_path, rows):
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="FictionalQA rewriting chains with Llama-3.1-70B via vLLM (AWQ INT4).")
    p.add_argument("--model", default=LLAMA_AWQ_MODEL_ID)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--n-fictions", type=int, default=100)
    p.add_argument("--styles", nargs="+", default=None,
                   choices=["news", "blog", "social", "corporate", "encyclopedia"])
    p.add_argument("--n-iterations", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    # E0 (≤~1084) + system+template (~200) + max-new-tokens (4096) ≈ 5400 → 8192.
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = p.parse_args()
    system_prompt = args.system_prompt if args.system_prompt else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    hf_login_if_token()
    random.seed(args.seed)

    print("\nLoading FictionalQA from HF Hub", flush=True)
    all_items = load_fictionalqa(allowed_styles=args.styles)
    questions = all_items[:1] if args.smoke_test else sample_items(all_items, args.n_fictions, args.seed)
    print(f"\nUsing {len(questions)} fictions", flush=True)

    done = load_done_keys(args.output)
    if done:
        print(f"\nResume: {len(done)} chains already in {args.output}", flush=True)

    total_chains = len(questions) * sum(len(pool) for pool in ALL_INSTRUCTIONS.values())
    print(f"\nPlan: {len(questions)} fictions × 4 × 3 = {total_chains} chains")

    llm, tokenizer = load_vllm(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    n_done = 0; n_to_do = total_chains - len(done); t_start = time.time()

    for q in questions:
        qid = q["id"]; E0 = q["text"]
        specs_to_run = []
        for (group, itype), pool in ALL_INSTRUCTIONS.items():
            for run, instruction in enumerate(pool):
                if (qid, group, itype, run) in done:
                    continue
                specs_to_run.append({
                    "group": group, "instruction_type": itype, "run": run, "instruction": instruction,
                })
        if not specs_to_run:
            continue

        t0 = time.time()
        specs_done = run_doc(
            llm, tokenizer, E0, specs_to_run,
            n_iterations=args.n_iterations, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens, system_prompt=system_prompt,
        )
        elapsed = time.time() - t0

        rows = []
        for s in specs_done:
            for step, text in enumerate(s["chain"]):
                rows.append({
                    "qid": qid, "question": q["question"],
                    "gold_answer": q["answer"], "gold_answer_aliases": ALIAS_SEP.join(q["aliases"]),
                    "style": q["style"], "event_id": q["event_id"],
                    "group": s["group"], "instruction_type": s["instruction_type"], "run": s["run"],
                    "instruction_used": s["instruction"] if step > 0 else "",
                    "step": step, "text": text,
                    "n_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                })
        append_rows(args.output, rows)
        n_done += len(specs_done)
        avg = (time.time() - t_start) / max(n_done, 1)
        remaining = (n_to_do - n_done) * avg
        print(f"[{n_done}/{n_to_do}] {qid} | {len(specs_done)} chains | {elapsed:.1f}s "
              f"| ETA {remaining/60:.1f} min", flush=True)

    print(f"\nDone. Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
