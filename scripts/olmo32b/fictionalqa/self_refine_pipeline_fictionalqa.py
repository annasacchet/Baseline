"""FictionalQA Self-Refine pipeline (RQ3) — OLMo-3.1-32B-Instruct (bf16) via vLLM.

Same Rewriter / Critic / Refiner loop as the MuSiQue self-refine pipeline,
operating on FictionalQA documents.

E0 is taken from the baseline rewriting CSV when present; otherwise rebuilt
via the HF loader in the baseline pipeline.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "olmo32b"))
from _common.olmo_constants import (  # noqa: E402
    ALIAS_SEP,
    ALL_INSTRUCTIONS,
    CHAIN_KEYS,
    OLMO_MODEL_ID,
    SELF_REFINE_CRITIC_PRIOR_BLOCK,
    SELF_REFINE_CRITIC_TEMPLATE,
    SELF_REFINE_REFINER_PRIOR_BLOCK,
    SELF_REFINE_REFINER_TEMPLATE,
    SELF_REFINE_REWRITER_TEMPLATE,
    render_prior_feedback,
)
from _common.olmo_vllm import (  # noqa: E402
    generate_batch_vllm,
    hf_login_if_token,
    load_vllm,
    render_chat,
)
from fictionalqa.rewriting_pipeline_fictionalqa import (  # noqa: E402
    load_fictionalqa, sample_items,
)

DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "olmo32b" / "fictionalqa" / "self_refine_chains_fictionalqa.csv"
DEFAULT_BASELINE_CSV = REPO_ROOT / "results" / "olmo32b" / "fictionalqa" / "rewriting_chains_fictionalqa.csv"


def load_e0_from_baseline(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    step0 = df[df["step"] == 0].drop_duplicates(subset=["qid"])
    result = {}
    for _, row in step0.iterrows():
        result[row["qid"]] = (
            row["question"],
            row.get("gold_answer", ""),
            row.get("gold_answer_aliases", "") or "",
            row.get("style", ""),
            row.get("event_id", ""),
            row["text"],
        )
    return result


def self_refine_step_batched(
    llm, tokenizer, specs, E0, *,
    rewriter_temperature, rewriter_max_new_tokens,
    critic_temperature, critic_max_new_tokens,
    refiner_temperature, refiner_max_new_tokens,
):
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

    for i in range(len(specs)):
        specs[i]["steps"].append({
            "E_draft": drafts[i], "feedback": feedbacks[i], "E_tilde": refined[i],
        })
        specs[i]["current"] = refined[i]
        specs[i]["prior_feedbacks"] = specs[i]["prior_feedbacks"] + [feedbacks[i]]


def load_done_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return {tuple(row[k] for k in CHAIN_KEYS) for _, row in df[CHAIN_KEYS].drop_duplicates().iterrows()}


def append_rows(csv_path: Path, rows: list):
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(),
                              index=False, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="FictionalQA self-refine chains with OLMo-3.1-32B via vLLM.")
    p.add_argument("--model", default=OLMO_MODEL_ID)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    p.add_argument("--n-fictions", type=int, default=100)
    p.add_argument("--styles", nargs="+", default=None,
                   choices=["news", "blog", "social", "corporate", "encyclopedia"])
    p.add_argument("--n-iterations", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=None)
    # Per-role temperatures inherited from scripts/15q/self_refine_pipeline.py.
    # See the rationale in scripts/olmo32b/musique/self_refine_pipeline_musique.py.
    p.add_argument("--rewriter-temperature", type=float, default=0.7)
    p.add_argument("--critic-temperature", type=float, default=0.0)
    p.add_argument("--refiner-temperature", type=float, default=0.3)
    p.add_argument("--rewriter-max-new-tokens", type=int, default=4096)
    p.add_argument("--critic-max-new-tokens", type=int, default=1024)
    p.add_argument("--refiner-max-new-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=12288)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--quantization", default=None,
                   help="vLLM quantization. Pass 'bitsandbytes' for on-the-fly NF4 4-bit.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    hf_login_if_token()
    random.seed(args.seed)

    e0_lookup: dict = {}
    if args.baseline_csv.exists():
        e0_lookup = load_e0_from_baseline(args.baseline_csv)
        print(f"Loaded E0 for {len(e0_lookup)} fictions from {args.baseline_csv}", flush=True)

    if args.smoke_test and e0_lookup:
        qids_to_run = [next(iter(e0_lookup))]
        print(f"*** SMOKE TEST: 1 fiction ({qids_to_run[0]}) ***", flush=True)
    elif e0_lookup and not args.smoke_test:
        qids_to_run = list(e0_lookup.keys())
        print(f"Using {len(qids_to_run)} fictions from baseline CSV", flush=True)
    else:
        all_items = load_fictionalqa(allowed_styles=args.styles)
        chosen = (all_items[:1] if args.smoke_test
                  else sample_items(all_items, args.n_fictions, args.seed))
        for it in chosen:
            e0_lookup[it["id"]] = (
                it["question"], it["answer"], ALIAS_SEP.join(it["aliases"]),
                it["style"], it["event_id"], it["text"],
            )
        qids_to_run = [it["id"] for it in chosen]
        print(f"Rebuilt E0 for {len(qids_to_run)} fictions from HF", flush=True)

    done = load_done_keys(args.output)
    if done:
        print(f"Resume: {len(done)} chains already in {args.output}", flush=True)

    n_runs_effective = (args.n_runs if args.n_runs is not None
                        else max(len(p) for p in ALL_INSTRUCTIONS.values()))
    total_chains = len(qids_to_run) * sum(
        min(len(pool), n_runs_effective) for pool in ALL_INSTRUCTIONS.values()
    )
    print(f"Plan: {len(qids_to_run)} fictions x 4 x {n_runs_effective} = {total_chains} chains, "
          f"each = {args.n_iterations} steps")

    llm, tokenizer = load_vllm(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        quantization=args.quantization,
    )

    n_done = 0; n_to_do = total_chains - len(done); t_start = time.time()

    for qid in qids_to_run:
        question_text, gold_answer, gold_aliases, style, event_id, E0 = e0_lookup[qid]

        specs = []
        for (group, itype), pool in ALL_INSTRUCTIONS.items():
            for run, instruction in enumerate(pool[:n_runs_effective]):
                if (qid, group, itype, run) in done:
                    continue
                specs.append({
                    "group": group, "instruction_type": itype, "run": run,
                    "instruction": instruction,
                    "current": E0, "steps": [], "prior_feedbacks": [],
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

        rows = []
        for s in specs:
            rows.append({
                "qid": qid, "question": question_text,
                "gold_answer": gold_answer, "gold_answer_aliases": gold_aliases,
                "style": style, "event_id": event_id,
                "group": s["group"], "instruction_type": s["instruction_type"], "run": s["run"],
                "instruction_used": "", "step": 0,
                "text": E0,
                "draft_text": "", "critic_feedback": "",
                "n_tokens": len(tokenizer.encode(E0, add_special_tokens=False)),
                "draft_n_tokens": 0,
            })
            for step_idx, st in enumerate(s["steps"], start=1):
                rows.append({
                    "qid": qid, "question": question_text,
                    "gold_answer": gold_answer, "gold_answer_aliases": gold_aliases,
                    "style": style, "event_id": event_id,
                    "group": s["group"], "instruction_type": s["instruction_type"], "run": s["run"],
                    "instruction_used": s["instruction"], "step": step_idx,
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
