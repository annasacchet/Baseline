"""NewsQA rewriting chain pipeline — OLMo-3.1-32B-Instruct (bf16) via vLLM.

Same structure as scripts/newsqa/rewriting_pipeline_newsqa.py, but the rewriter
is OLMo-3.1-32B-Instruct served by vLLM with tensor parallelism, and the prompt
uses the validated XML + system-prompt fix.

Picks one validated question per NewsQA story (see the original docstring for
the agreement rule). Stores gold answer aliases ||-joined for SQuAD-style F1.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gptoss120b"))
from _common.olmo_constants import (  # noqa: E402
    ALIAS_SEP,
    ALL_INSTRUCTIONS,
    CHAIN_KEYS,
    DEFAULT_SYSTEM_PROMPT,
    OLMO_MODEL_ID,
    REWRITE_TEMPLATE,
)
from _common.olmo_vllm import (  # noqa: E402
    generate_batch_vllm,
    hf_login_if_token,
    load_vllm,
    render_chat,
)

DEFAULT_DATASET_PATH = Path(os.environ.get(
    "NEWSQA_DATASET",
    "/mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv",
))
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "gptoss120b" / "newsqa" / "rewriting_chains_newsqa.csv"


# ---------------------------------------------------------------------------
# NewsQA dataset loading (same logic as scripts/newsqa/rewriting_pipeline_newsqa.py)
# ---------------------------------------------------------------------------

def _span_to_text(span_key, story_text):
    parts = []
    try:
        for span in span_key.split(","):
            s_str, e_str = span.split(":")
            parts.append(story_text[int(s_str):int(e_str)])
    except (ValueError, IndexError):
        return None
    out = " ".join(p.strip() for p in parts if p.strip()).strip()
    return out or None


def parse_validated_answers(value, story_text):
    if not isinstance(value, str) or not value.strip():
        return [], 0, 0
    try:
        d = json.loads(value)
    except json.JSONDecodeError:
        return [], 0, 0
    answers, n_none, n_bad = [], 0, 0
    for key, count in d.items():
        if key == "none":
            n_none = int(count); continue
        if key == "badQuestion":
            n_bad = int(count); continue
        ans = _span_to_text(key, story_text)
        if ans:
            answers.append((ans, int(count)))
    answers.sort(key=lambda t: -t[1])
    return answers, n_none, n_bad


def parse_sourcer_agreement(value, story_text):
    if not isinstance(value, str) or not value.strip():
        return [], 0
    counts, n_none = {}, 0
    for raw in value.split("|"):
        raw = raw.strip()
        if not raw or raw == "None":
            n_none += 1 if raw == "None" else 0
            continue
        ans = _span_to_text(raw, story_text)
        if ans:
            counts[ans] = counts.get(ans, 0) + 1
    return sorted(counts.items(), key=lambda t: -t[1]), n_none


def pick_best_question_for_story(rows):
    candidates = []
    for r in rows:
        try:
            if float(r.get("is_answer_absent") or 0) >= 0.5: continue
            if float(r.get("is_question_bad") or 0) >= 0.5: continue
        except (TypeError, ValueError):
            continue
        story_text = r.get("story_text") or ""
        if not story_text:
            continue
        v_spans, v_none, v_bad = parse_validated_answers(r.get("validated_answers"), story_text)
        v_agreed = [(ans, c) for ans, c in v_spans if c >= 2]
        if v_agreed and v_agreed[0][1] > max(v_none, v_bad):
            candidates.append({
                "id": r["story_id"], "text": story_text, "question": r["question"],
                "answer": v_agreed[0][0], "aliases": sorted({a for a, _ in v_agreed}),
            })
            continue
        s_spans, s_none = parse_sourcer_agreement(r.get("answer_char_ranges"), story_text)
        s_agreed = [(ans, c) for ans, c in s_spans if c >= 2]
        if s_agreed and s_agreed[0][1] > s_none:
            candidates.append({
                "id": r["story_id"], "text": story_text, "question": r["question"],
                "answer": s_agreed[0][0], "aliases": sorted({a for a, _ in s_agreed}),
            })
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-len(c["aliases"]), -len(c["question"])))
    return candidates[0]


def load_newsqa(path: Path):
    print("  reading CSV (this may take a minute)...", flush=True)
    df = pd.read_csv(
        path,
        dtype={"story_id": "string", "question": "string", "answer_char_ranges": "string",
               "validated_answers": "string", "story_text": "string"},
        keep_default_na=False,
    )
    print(f"  CSV loaded: {len(df):,} rows · {df['story_id'].nunique():,} stories", flush=True)
    items = []
    for _, story_rows in df.groupby("story_id", sort=False):
        rec = pick_best_question_for_story(story_rows.to_dict(orient="records"))
        if rec is not None:
            items.append(rec)
    print(f"  answerable stories after quality filter: {len(items):,}", flush=True)
    return items


def sample_items(items, n, seed):
    rng = random.Random(seed); pool = list(items); rng.shuffle(pool); return pool[:n]


# ---------------------------------------------------------------------------
# Chain runner — vLLM batched generation
# ---------------------------------------------------------------------------

def run_story(llm, tokenizer, E0, specs, *, n_iterations, temperature,
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
    p = argparse.ArgumentParser(description="NewsQA rewriting chains with OLMo-3.1-32B via vLLM.")
    p.add_argument("--model", default=OLMO_MODEL_ID)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--n-questions", type=int, default=100)
    p.add_argument("--n-iterations", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    # NewsQA articles are long (~30+ sentences); keep 4096 headroom.
    p.add_argument("--max-new-tokens", type=int, default=4096)
    # NewsQA articles + 4096 generation can blow past 8192 — use 12k window.
    p.add_argument("--max-model-len", type=int, default=12288)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = p.parse_args()
    system_prompt = args.system_prompt if args.system_prompt else None

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    hf_login_if_token()
    random.seed(args.seed)

    print(f"\nLoading NewsQA from {args.dataset}", flush=True)
    all_items = load_newsqa(args.dataset)

    questions = all_items[:1] if args.smoke_test else sample_items(all_items, args.n_questions, args.seed)
    print(f"\nUsing {len(questions)} stories", flush=True)

    done = load_done_keys(args.output)
    if done:
        print(f"\nResume: {len(done)} chains already in {args.output}", flush=True)

    total_chains = len(questions) * sum(len(pool) for pool in ALL_INSTRUCTIONS.values())
    print(f"\nPlan: {len(questions)} stories x 4 x 3 = {total_chains} chains")

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
        specs_done = run_story(
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
