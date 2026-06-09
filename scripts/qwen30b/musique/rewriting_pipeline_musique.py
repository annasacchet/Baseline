"""MuSiQue rewriting chain pipeline — OLMo-3.1-32B-Instruct (bf16) via vLLM.

Mirrors scripts/15q/rewriting_pipeline_optimized.py (XML prompt + system prompt
fix validated by project_rewriting_prompt_fix), retargeted to OLMo-3.1-32B-Instruct
served by vLLM with tensor parallelism across the two GPUs on Homer.

Why vLLM:
  - bf16 OLMo-32B is ~64 GB of weights — vLLM with TP=2 splits each layer
    across both GPUs so both are active on every forward.
  - Paged attention + continuous batching gives 10-15x throughput vs raw
    HF transformers on 2x consumer GPUs.

Output schema is unchanged:
  qid, question, group, instruction_type, run, instruction_used, step, text,
  n_tokens
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "qwen30b"))
from _common.olmo_constants import (  # noqa: E402
    ALL_INSTRUCTIONS,
    CHAIN_KEYS,
    DEFAULT_SYSTEM_PROMPT,
    OLMO_MODEL_ID,
    REWRITE_TEMPLATE,
)
def _load_backend(name):
    """Return (load_vllm, render_chat, generate_batch_vllm, hf_login_if_token)
    from the vLLM or the HF backend. Both expose the same names."""
    if name == "hf":
        from _common.olmo_hf import (
            generate_batch_vllm, hf_login_if_token, load_vllm, render_chat,
        )
    else:
        from _common.olmo_vllm import (
            generate_batch_vllm, hf_login_if_token, load_vllm, render_chat,
        )
    return load_vllm, render_chat, generate_batch_vllm, hf_login_if_token

DEFAULT_DATASET_PATH = Path(os.environ.get(
    "MUSIQUE_DATASET",
    str(REPO_ROOT / "musique_ans_v1.0_dev.jsonl"),
))
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "qwen30b" / "musique" / "rewriting_chains_musique.csv"


def load_musique(path: Path) -> list:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def hop_count(item: dict) -> int:
    qid = item.get("id", "")
    m = re.match(r"(\d+)hop__", qid)
    if m:
        return int(m.group(1))
    return len(item.get("question_decomposition", []))


def build_E0(item: dict, only_supporting: bool) -> str:
    paragraphs = item["paragraphs"]
    if only_supporting:
        paragraphs = [p for p in paragraphs if p.get("is_supporting")]
        paragraphs.sort(key=lambda p: p.get("idx", 0))
    return "\n\n".join(f"{p['title']}. {p['paragraph_text']}" for p in paragraphs)


def balance_by_hop(items: list, n_per_hop: int, seed: int) -> list:
    rng = random.Random(seed)
    by_hop = defaultdict(list)
    for it in items:
        by_hop[hop_count(it)].append(it)
    balanced = []
    for h in (2, 3, 4):
        pool = list(by_hop[h])
        rng.shuffle(pool)
        balanced.extend(pool[: min(n_per_hop, len(pool))])
    return balanced


# ---------------------------------------------------------------------------
# Chain runner — vLLM batched generation
# ---------------------------------------------------------------------------

def run_question(llm, tokenizer, E0, specs, *, n_iterations, temperature,
                 max_new_tokens, system_prompt):
    """Advance all chains of one question step-by-step, batched together via vLLM."""
    for s in specs:
        s["chain"] = [E0]
        s["current"] = E0
    for it in range(n_iterations):
        print(f"  [step {it + 1}/{n_iterations}] rewriting {len(specs)} chains ...", flush=True)
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
            s["chain"].append(new_text)
            s["current"] = new_text
    return specs


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def load_done_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return {tuple(row[k] for k in CHAIN_KEYS) for _, row in df[CHAIN_KEYS].drop_duplicates().iterrows()}


def append_rows(csv_path: Path, rows: list):
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="MuSiQue rewriting chains with OLMo-3.1-32B via vLLM.")
    p.add_argument("--model", default=OLMO_MODEL_ID)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--n-per-hop", type=int, default=200)
    p.add_argument("--n-iterations", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="vLLM context window (prompt+output). 8192 is enough for MuSiQue.")
    p.add_argument("--gpu-mem-util", type=float, default=0.90,
                   help="Fraction of each GPU vLLM may use. Lower if OOM during load.")
    p.add_argument("--tensor-parallel-size", type=int, default=None,
                   help="Default: number of visible CUDA devices.")
    p.add_argument("--quantization", default=None,
                   help="vLLM quantization. Pass 'bitsandbytes' for on-the-fly NF4 4-bit.")
    p.add_argument("--enforce-eager", action="store_true",
                   help="Disable CUDA graph capture / torch.compile (avoids JIT C++ build failures).")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                   help="Inference backend. 'hf' = transformers 4-bit (robust on old CUDA drivers).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--qids-file", type=Path, default=None)
    p.add_argument("--only-supporting", action="store_true")
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
                   help="Pass '' to disable the system prompt.")
    args = p.parse_args()

    # Bind backend functions as module globals so run_question() can use them.
    global load_vllm, render_chat, generate_batch_vllm, hf_login_if_token
    load_vllm, render_chat, generate_batch_vllm, hf_login_if_token = _load_backend(args.backend)
    print(f"Backend: {args.backend}", flush=True)

    system_prompt = args.system_prompt if args.system_prompt else None

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    hf_login_if_token()
    random.seed(args.seed)

    print(f"\nLoading MuSiQue from {args.dataset}", flush=True)
    raw = load_musique(args.dataset)
    print(f"  loaded {len(raw)} items", flush=True)

    if args.qids_file:
        wanted = set()
        with open(args.qids_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    wanted.add(line)
        questions = [it for it in raw if it["id"] in wanted]
        print(f"\n*** PINNED to {len(questions)}/{len(wanted)} qids ***", flush=True)
    elif args.smoke_test:
        questions = [it for it in raw if it["id"] == "2hop__635544_110949"]
        if not questions:
            print("ERROR: pilot question not found", file=sys.stderr)
            sys.exit(1)
        print("\n*** SMOKE TEST: 1 question (pilot 2-hop) ***", flush=True)
    else:
        questions = balance_by_hop(raw, args.n_per_hop, args.seed)
        print(f"\nUsing {len(questions)} questions (balanced by hop)", flush=True)

    done = load_done_keys(args.output)
    if done:
        print(f"\nResume: {len(done)} chains already in {args.output}", flush=True)

    total_chains = len(questions) * sum(len(pool) for pool in ALL_INSTRUCTIONS.values())
    print(f"\nSystem prompt: {'ON' if system_prompt else 'OFF'}", flush=True)
    print(f"Plan: {len(questions)} questions x 4 instructions x 3 wordings = {total_chains} chains")

    llm, tokenizer = load_vllm(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        quantization=args.quantization,
        enforce_eager=args.enforce_eager,
    )

    n_done = 0
    n_to_do = total_chains - len(done)
    t_start = time.time()

    for q in questions:
        qid = q["id"]
        question_text = q["question"]
        E0 = build_E0(q, only_supporting=args.only_supporting)

        specs_to_run = []
        for (group, instruction_type), pool in ALL_INSTRUCTIONS.items():
            for run, instruction in enumerate(pool):
                if (qid, group, instruction_type, run) in done:
                    continue
                specs_to_run.append({
                    "group": group,
                    "instruction_type": instruction_type,
                    "run": run,
                    "instruction": instruction,
                })
        if not specs_to_run:
            continue

        q_idx = questions.index(q) + 1
        print(f"\n[Q {q_idx}/{len(questions)}] {qid} — {len(specs_to_run)} chains to run ...", flush=True)
        t0 = time.time()
        specs_done = run_question(
            llm, tokenizer, E0, specs_to_run,
            n_iterations=args.n_iterations,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            system_prompt=system_prompt,
        )
        elapsed = time.time() - t0

        rows = []
        for s in specs_done:
            for step, text in enumerate(s["chain"]):
                rows.append({
                    "qid": qid,
                    "question": question_text,
                    "group": s["group"],
                    "instruction_type": s["instruction_type"],
                    "run": s["run"],
                    "instruction_used": s["instruction"] if step > 0 else "",
                    "step": step,
                    "text": text,
                    "n_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                })
        append_rows(args.output, rows)
        n_done += len(specs_done)

        avg = (time.time() - t_start) / max(n_done, 1)
        remaining = (n_to_do - n_done) * avg
        print(
            f"[{n_done}/{n_to_do}] {qid} | {len(specs_done)} chains | {elapsed:.1f}s "
            f"| ETA {remaining/60:.1f} min", flush=True,
        )

    print(f"\nDone. Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
