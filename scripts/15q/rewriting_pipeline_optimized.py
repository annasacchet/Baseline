"""
Rewriting pipeline — OPTIMIZED GPU version.

Same pipeline, prompts, output schema and resume logic as
scripts/15q/rewriting_pipeline.py. The ONLY change is how the rewriter is
fed to the GPU:

  * For each MuSiQue question, the 12 chains (4 instructions × 3 wordings)
    are advanced step-by-step IN PARALLEL via batched generation.
    Step 1 of all 12 chains -> one forward pass, step 2 of all 12 chains
    -> one forward pass, etc.
  * padding_side='left' + padding=True so decoder-only batched generate
    works correctly.
  * KV-cache explicitly enabled; torch.inference_mode() instead of no_grad.
  * --batch-size flag caps how many chains are batched at once (default 12 =
    all chains of a question).

Outputs are unchanged: same rows, same columns, same CSV path.
"""

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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = REPO_ROOT / "musique_ans_v1.0_dev.jsonl"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "rewriting_chains.csv"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]


ALL_INSTRUCTIONS = {
    ("style", "formality"): [
        "Make the text more formal.",
        "Rephrase it to be more formal.",
        "Too conversational, rephrase it to be more formal.",
    ],
    ("style", "paraphrase"): [
        "Paraphrase this.",
        "Reword this text.",
        "Use different wording.",
    ],
    ("content", "shorten"): [
        "Make wording more concise.",
        "Rephrase for clarity and conciseness.",
        "Improve accuracy, clarity, and conciseness of language.",
    ],
    ("content", "elaborate"): [
        "Elaborate on the content, adding relevant details while staying faithful to the source text.",
        "Expand the text with more context, without introducing information that is not supported by the original.",
        "Add more detail, keeping every fact grounded in the source material.",
    ],
}

REWRITE_TEMPLATE = """You are a precise text rewriting assistant. Your task is to rewrite the text provided inside the XML tags according to the specific instruction.

<source_text>
{text}
</source_text>

Instruction: {instruction}

Strict Rule: Return ONLY the rewritten text. Do not include any preamble, introduction, markdown formatting outside the text, or commentary."""


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful text rewriting assistant. "
    "When the user provides a text and an instruction, you must rewrite the "
    "ENTIRE text according to the instruction. "
    "The source text may contain multiple independent paragraphs separated by "
    "blank lines; you MUST rewrite every single paragraph, in the same order, "
    "without omitting, merging, or summarizing any of them. "
    "Preserve the original number of paragraphs and the factual content of each. "
    "Never answer questions about the text — only rewrite it. "
    "Return only the rewritten text, with no preamble or commentary."
)


# ---------------------------------------------------------------------------
# Dataset loading
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


# ---------------------------------------------------------------------------
# Model loading + BATCHED generation
# ---------------------------------------------------------------------------

def load_rewriter(model_id: str, use_4bit: bool = False):
    print(f"Loading rewriter: {model_id} (4-bit={use_4bit})", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Required for decoder-only batched generation.
    tok.padding_side = "left"
    kwargs = {"device_map": "auto", "trust_remote_code": True}
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
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
    print(f"  device map: {getattr(model, 'hf_device_map', 'n/a')}", flush=True)
    return tok, model


def _format_prompt(tokenizer, user_prompt: str, system_prompt: str | None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (system_prompt + "\n\n" if system_prompt else "") + user_prompt


@torch.inference_mode()
def generate_batch(
    tokenizer, model, user_prompts: list[str], *,
    temperature: float, max_new_tokens: int, system_prompt: str | None,
) -> list[str]:
    """Batched generation. Returns one decoded string per input prompt."""
    if not user_prompts:
        return []
    texts = [_format_prompt(tokenizer, up, system_prompt) for up in user_prompts]
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=False,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    out = model.generate(**enc, **gen_kwargs)
    gen = out[:, enc["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def run_question_batched(
    tokenizer, model, E0: str, all_chains_specs: list[dict],
    *, n_iterations: int, temperature: float, max_new_tokens: int,
    system_prompt: str | None, batch_size: int,
):
    """Advance many chains for the same question in lockstep, batched per step.

    `all_chains_specs` is a list of dicts {group, instruction_type, run,
    instruction} — one per chain to compute. Returns the same list enriched
    with `chain` = [E0, E1, ..., En] (one chain per spec).
    """
    # Initial state: every chain starts at E0.
    for spec in all_chains_specs:
        spec["chain"] = [E0]
        spec["current"] = E0

    for _ in range(n_iterations):
        # Build all user prompts for this step.
        prompts = [
            REWRITE_TEMPLATE.format(instruction=s["instruction"], text=s["current"])
            for s in all_chains_specs
        ]
        # Chunked batched generation.
        outs = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            outs.extend(generate_batch(
                tokenizer, model, chunk,
                temperature=temperature, max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
            ))
        for s, new_text in zip(all_chains_specs, outs):
            s["chain"].append(new_text)
            s["current"] = new_text

    return all_chains_specs


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
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate rewriting chains on GPU (optimized, batched).")
    parser.add_argument("--model", default="allenai/OLMo-3.1-32B-Instruct")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--n-per-hop", type=int, default=405)
    parser.add_argument("--n-iterations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--qids-file", type=Path, default=None)
    parser.add_argument("--only-supporting", action="store_true")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
                        help="Pass '' to disable the system prompt.")
    parser.add_argument(
        "--batch-size", type=int, default=12,
        help="Number of chains advanced in parallel per generate() call "
             "(default: 12 = all chains of one question). Lower if you OOM.",
    )
    args = parser.parse_args()
    system_prompt = args.system_prompt if args.system_prompt else None

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("HF login OK", flush=True)
    else:
        print("HF_TOKEN not set — proceeding without login", flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\nLoading MuSiQue from {args.dataset}", flush=True)
    raw = load_musique(args.dataset)
    print(f"  loaded {len(raw)} items", flush=True)
    print(f"  hop distribution: {dict((h, sum(1 for it in raw if hop_count(it)==h)) for h in (2,3,4))}", flush=True)

    if args.qids_file:
        wanted = set()
        with open(args.qids_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    wanted.add(line)
        questions = [it for it in raw if it["id"] in wanted]
        missing = wanted - {q["id"] for q in questions}
        if missing:
            print(f"WARNING: {len(missing)} qids from {args.qids_file} not found:", file=sys.stderr)
            for m in sorted(missing):
                print(f"  {m}", file=sys.stderr)
        if not questions:
            print(f"ERROR: no qids matched", file=sys.stderr)
            sys.exit(1)
        print(f"\n*** PINNED to {len(questions)}/{len(wanted)} qids ***", flush=True)
    elif args.smoke_test:
        questions = [it for it in raw if it["id"] == "2hop__635544_110949"]
        if not questions:
            print("ERROR: pilot question not found", file=sys.stderr)
            sys.exit(1)
        print(f"\n*** SMOKE TEST: 1 question (pilot 2-hop) ***", flush=True)
    else:
        questions = balance_by_hop(raw, args.n_per_hop, args.seed)
        print(f"\nUsing {len(questions)} questions, balanced across hop counts", flush=True)

    done = load_done_keys(args.output)
    if done:
        print(f"\nResume: {len(done)} chains already in {args.output}", flush=True)

    total_chains = (
        len(questions) * sum(len(pool) for pool in ALL_INSTRUCTIONS.values())
    )
    print(f"\nSystem prompt: {'ON' if system_prompt else 'OFF'}", flush=True)
    print(f"Plan: {len(questions)} questions x 4 instructions x 3 wordings = {total_chains} chains")
    print(f"Batch size: {args.batch_size}")

    tokenizer, model = load_rewriter(args.model, use_4bit=args.use_4bit)

    n_done = 0
    n_to_do = total_chains - len(done)
    t_start = time.time()

    for q in questions:
        qid = q["id"]
        question_text = q["question"]
        E0 = build_E0(q, only_supporting=args.only_supporting)

        # Gather all chains still to be done for this question.
        specs_to_run = []
        for (group, instruction_type), pool in ALL_INSTRUCTIONS.items():
            for run, instruction in enumerate(pool):
                key = (qid, group, instruction_type, run)
                if key in done:
                    continue
                specs_to_run.append({
                    "group": group,
                    "instruction_type": instruction_type,
                    "run": run,
                    "instruction": instruction,
                })

        if not specs_to_run:
            continue

        t0 = time.time()
        specs_done = run_question_batched(
            tokenizer, model, E0, specs_to_run,
            n_iterations=args.n_iterations,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            system_prompt=system_prompt,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - t0

        # Persist all chains of this question.
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
            f"[{n_done}/{n_to_do}] {qid} | {len(specs_done)} chains in "
            f"{elapsed:.1f}s | ETA {remaining/60:.1f} min",
            flush=True,
        )

    print(f"\nDone. Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
