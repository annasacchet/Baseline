"""MuSiQue Self-Refine pipeline (RQ3) — Llama-3.1-70B-Instruct-AWQ-INT4 via vLLM.

Generates rewriting chains E_0 → Ẽ_1 → Ẽ_2 → Ẽ_3 using a Rewriter / Critic /
Refiner loop where all three roles are the same Llama-70B model.

Pipeline (per question, per (group, instruction_type), per wording = run)
-------------------------------------------------------------------------
At each step t:
  1. Rewriter  — receives Ẽ_{t-1} and the instruction → produces E_draft.
  2. Critic    — receives E_0 and E_draft → identifies facts that were changed,
                 removed, or distorted. Output: Verdict + Issues block.
  3. Refiner   — receives E_draft and the feedback → produces Ẽ_t, preserving
                 style but fixing factual errors. Ẽ_t feeds step t+1.

Why vLLM here
-------------
Self-refine triples the per-step compute (3 generations per step instead of 1)
and the rewriter+refiner outputs are long. vLLM with TP=2 + AWQ-INT4 + paged
attention is the only practical way to run this on 2x A6000 in a reasonable
time. Per question we batch all 12 chains together for each of the three
phases, giving vLLM enough work to keep both GPUs busy.

Output schema (extends rewriting_chains_musique.csv)
----------------------------------------------------
qid, question, group, instruction_type, run, instruction_used, step, text,
n_tokens, draft_text, critic_feedback, draft_n_tokens
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
sys.path.insert(0, str(REPO_ROOT / "scripts" / "llama70b"))
from _common.llama_constants import (  # noqa: E402
    ALL_INSTRUCTIONS,
    CHAIN_KEYS,
    LLAMA_AWQ_MODEL_ID,
    SELF_REFINE_CRITIC_PRIOR_BLOCK,
    SELF_REFINE_CRITIC_TEMPLATE,
    SELF_REFINE_REFINER_PRIOR_BLOCK,
    SELF_REFINE_REFINER_TEMPLATE,
    SELF_REFINE_REWRITER_TEMPLATE,
    render_prior_feedback,
)
from _common.llama_vllm import (  # noqa: E402
    generate_batch_vllm,
    hf_login_if_token,
    load_vllm,
    render_chat,
)

DEFAULT_DATASET_PATH = Path(os.environ.get(
    "MUSIQUE_DATASET",
    str(REPO_ROOT / "musique_ans_v1.0_dev.jsonl"),
))
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "llama70b" / "musique" / "self_refine_chains_musique.csv"
DEFAULT_BASELINE_CSV = REPO_ROOT / "results" / "llama70b" / "musique" / "rewriting_chains_musique.csv"


# ---------------------------------------------------------------------------
# MuSiQue helpers (same as the baseline rewriting pipeline)
# ---------------------------------------------------------------------------

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


def load_e0_from_baseline(csv_path: Path) -> dict:
    """Return {qid: (question_text, E0_text)} from step=0 rows of the baseline CSV.

    Using the baseline rewriting CSV guarantees byte-identical E0 between RQ1
    (rewriting_pipeline) and RQ3 (self_refine_pipeline).
    """
    df = pd.read_csv(csv_path)
    step0 = df[df["step"] == 0].drop_duplicates(subset=["qid"])
    return {row["qid"]: (row["question"], row["text"]) for _, row in step0.iterrows()}


# ---------------------------------------------------------------------------
# Self-refine — one iteration, batched across all chains of one question
# ---------------------------------------------------------------------------

def self_refine_step_batched(
    llm, tokenizer, specs, E0, *,
    rewriter_temperature, rewriter_max_new_tokens,
    critic_temperature, critic_max_new_tokens,
    refiner_temperature, refiner_max_new_tokens,
):
    """One self-refine iteration over all `specs` of a single question.

    Each spec carries: instruction, current text, prior_feedbacks (list).
    Mutates the specs in place by appending step results.
    """
    n = len(specs)

    # 1. Rewriter — one prompt per spec
    rewriter_prompts = [
        render_chat(
            tokenizer, None,
            SELF_REFINE_REWRITER_TEMPLATE.format(
                instruction=s["instruction"], text=s["current"],
            ),
        )
        for s in specs
    ]
    drafts = generate_batch_vllm(
        llm, rewriter_prompts,
        temperature=rewriter_temperature, max_new_tokens=rewriter_max_new_tokens,
    )

    # 2. Critic — uses E0, draft, and the per-spec prior feedback history.
    critic_prompts = []
    for s, draft in zip(specs, drafts):
        prior = render_prior_feedback(s["prior_feedbacks"], SELF_REFINE_CRITIC_PRIOR_BLOCK)
        critic_prompts.append(render_chat(
            tokenizer, None,
            SELF_REFINE_CRITIC_TEMPLATE.format(
                original=E0, draft=draft, prior_feedback_block=prior,
            ),
        ))
    feedbacks = generate_batch_vllm(
        llm, critic_prompts,
        temperature=critic_temperature, max_new_tokens=critic_max_new_tokens,
    )

    # 3. Refiner
    refiner_prompts = []
    for s, draft, feedback in zip(specs, drafts, feedbacks):
        prior = render_prior_feedback(s["prior_feedbacks"], SELF_REFINE_REFINER_PRIOR_BLOCK)
        refiner_prompts.append(render_chat(
            tokenizer, None,
            SELF_REFINE_REFINER_TEMPLATE.format(
                draft=draft, feedback=feedback, prior_feedback_block=prior,
            ),
        ))
    refined = generate_batch_vllm(
        llm, refiner_prompts,
        temperature=refiner_temperature, max_new_tokens=refiner_max_new_tokens,
    )

    # Update specs in place
    for i in range(n):
        specs[i]["steps"].append({
            "E_draft": drafts[i],
            "feedback": feedbacks[i],
            "E_tilde": refined[i],
        })
        specs[i]["current"] = refined[i]
        specs[i]["prior_feedbacks"] = specs[i]["prior_feedbacks"] + [feedbacks[i]]


# ---------------------------------------------------------------------------
# Resume / IO
# ---------------------------------------------------------------------------

def load_done_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return {tuple(row[k] for k in CHAIN_KEYS) for _, row in df[CHAIN_KEYS].drop_duplicates().iterrows()}


def append_rows(csv_path: Path, rows: list):
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(),
                              index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="MuSiQue self-refine chains with Llama-3.1-70B-AWQ via vLLM.")
    p.add_argument("--model", default=LLAMA_AWQ_MODEL_ID)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV,
                   help="Baseline rewriting CSV used to source E0. If missing, "
                        "E0 is rebuilt from --dataset.")
    p.add_argument("--n-per-hop", type=int, default=200)
    p.add_argument("--n-iterations", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=None,
                   help="Limit to the first N wordings per instruction type (1-3).")
    # Per-role temperatures inherited from scripts/15q/self_refine_pipeline.py.
    # Rewriter=0.7: matches the baseline rewriting pipeline; on long contexts
    #   (20-paragraph MuSiQue E0) greedy decoding causes summarization rather
    #   than full-paragraph rewriting. The Self-Refine paper (Madaan et al.
    #   2023, §3.1) reports 0.7 for all roles, but our task is fact-checking
    #   on long contexts, not the dialogue/code/math tasks of the paper.
    # Critic=0.0: deterministic fact-checking judgement (reproducible across
    #   runs). Critic is a discriminative SUPPORTED/NOT_SUPPORTED task here.
    # Refiner=0.3: corrections are guided by the feedback (a strong prior);
    #   low but non-zero sampling avoids the long-context greedy-collapse
    #   while staying close to the feedback's directives.
    p.add_argument("--rewriter-temperature", type=float, default=0.7)
    p.add_argument("--critic-temperature", type=float, default=0.0)
    p.add_argument("--refiner-temperature", type=float, default=0.3)
    p.add_argument("--rewriter-max-new-tokens", type=int, default=2048)
    p.add_argument("--critic-max-new-tokens", type=int, default=1024)
    p.add_argument("--refiner-max-new-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--qids-file", type=Path, default=None)
    p.add_argument("--only-supporting", action="store_true")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    hf_login_if_token()
    random.seed(args.seed)

    # Source of E0: prefer the baseline CSV (byte-identical to RQ1).
    e0_lookup: dict = {}
    if args.baseline_csv.exists():
        e0_lookup = load_e0_from_baseline(args.baseline_csv)
        print(f"Loaded E0 for {len(e0_lookup)} questions from {args.baseline_csv}", flush=True)

    if args.qids_file:
        qids_to_run = [q.strip() for q in args.qids_file.read_text().splitlines() if q.strip()]
        print(f"Using {len(qids_to_run)} qids from {args.qids_file}", flush=True)
    elif args.smoke_test:
        qids_to_run = ["2hop__635544_110949"]
        print("*** SMOKE TEST: 1 question (pilot 2-hop) ***", flush=True)
    else:
        if not args.dataset.exists():
            print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
            sys.exit(1)
        raw = load_musique(args.dataset)
        qids_to_run = [it["id"] for it in balance_by_hop(raw, args.n_per_hop, args.seed)]
        print(f"Using {len(qids_to_run)} questions (balanced by hop)", flush=True)

    # Fill any missing E0 by rebuilding from MuSiQue.
    needs_rebuild = [qid for qid in qids_to_run if qid not in e0_lookup]
    if needs_rebuild:
        if not args.dataset.exists():
            print(f"ERROR: dataset not found ({args.dataset}) but {len(needs_rebuild)} qids "
                  f"are missing from baseline CSV.", file=sys.stderr)
            sys.exit(1)
        raw_index = {it["id"]: it for it in load_musique(args.dataset)}
        for qid in needs_rebuild:
            if qid not in raw_index:
                print(f"ERROR: qid {qid} missing from both baseline CSV and MuSiQue.",
                      file=sys.stderr)
                sys.exit(1)
            it = raw_index[qid]
            e0_lookup[qid] = (it["question"], build_E0(it, only_supporting=args.only_supporting))
        print(f"Rebuilt E0 for {len(needs_rebuild)} qids from MuSiQue.", flush=True)

    done = load_done_keys(args.output)
    if done:
        print(f"Resume: {len(done)} chains already in {args.output}", flush=True)

    n_runs_effective = (args.n_runs if args.n_runs is not None
                        else max(len(p) for p in ALL_INSTRUCTIONS.values()))
    total_chains = len(qids_to_run) * sum(
        min(len(pool), n_runs_effective) for pool in ALL_INSTRUCTIONS.values()
    )
    print(f"Plan: {len(qids_to_run)} questions x 4 instructions x {n_runs_effective} wordings "
          f"= {total_chains} chains, each = {args.n_iterations} self-refine steps")

    llm, tokenizer = load_vllm(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    n_done = 0
    n_to_do = total_chains - len(done)
    t_start = time.time()

    for qid in qids_to_run:
        question_text, E0 = e0_lookup[qid]

        # Build spec list for this question (skip chains already in CSV).
        specs = []
        for (group, itype), pool in ALL_INSTRUCTIONS.items():
            for run, instruction in enumerate(pool[:n_runs_effective]):
                if (qid, group, itype, run) in done:
                    continue
                specs.append({
                    "group": group, "instruction_type": itype, "run": run,
                    "instruction": instruction,
                    "current": E0,
                    "steps": [],
                    "prior_feedbacks": [],
                })
        if not specs:
            continue

        t0 = time.time()
        for _ in range(args.n_iterations):
            self_refine_step_batched(
                llm, tokenizer, specs, E0,
                rewriter_temperature=args.rewriter_temperature,
                rewriter_max_new_tokens=args.rewriter_max_new_tokens,
                critic_temperature=args.critic_temperature,
                critic_max_new_tokens=args.critic_max_new_tokens,
                refiner_temperature=args.refiner_temperature,
                refiner_max_new_tokens=args.refiner_max_new_tokens,
            )
        elapsed = time.time() - t0

        # Emit rows: step 0 = E0, then one per iteration.
        rows = []
        for s in specs:
            rows.append({
                "qid": qid, "question": question_text,
                "group": s["group"], "instruction_type": s["instruction_type"], "run": s["run"],
                "instruction_used": "",
                "step": 0,
                "text": E0,
                "draft_text": "",
                "critic_feedback": "",
                "n_tokens": len(tokenizer.encode(E0, add_special_tokens=False)),
                "draft_n_tokens": 0,
            })
            for step_idx, st in enumerate(s["steps"], start=1):
                rows.append({
                    "qid": qid, "question": question_text,
                    "group": s["group"], "instruction_type": s["instruction_type"], "run": s["run"],
                    "instruction_used": s["instruction"],
                    "step": step_idx,
                    "text": st["E_tilde"],
                    "draft_text": st["E_draft"],
                    "critic_feedback": st["feedback"],
                    "n_tokens": len(tokenizer.encode(st["E_tilde"], add_special_tokens=False)),
                    "draft_n_tokens": len(tokenizer.encode(st["E_draft"], add_special_tokens=False)),
                })
        append_rows(args.output, rows)
        n_done += len(specs)

        avg = (time.time() - t_start) / max(n_done, 1)
        remaining = (n_to_do - n_done) * avg
        print(f"[{n_done}/{n_to_do}] {qid} | {len(specs)} chains | {elapsed:.1f}s "
              f"| ETA {remaining/60:.1f} min", flush=True)

    print(f"\nDone. Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
