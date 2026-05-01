"""
Rewriting pipeline — PAU variant (Laban et al., 2024).

Same as rewriting_pipeline.py but adds --n-repetitions: each
(qid, group, instruction_type, wording) is run n times with temperature > 0
to produce the distribution needed for P/A/U statistics.

Output schema: same as rewriting_chains*.csv, plus:
  - repetition : int, 0-indexed repetition index (0 = first run)

CHAIN_KEYS = [qid, group, instruction_type, run, repetition]
where run = wording index (0/1/2) and repetition = stochastic repeat (0..n-1).
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET_PATH = REPO_ROOT / "musique_ans_v1.0_dev.jsonl"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "rewriting_chains_pau.csv"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run", "repetition"]


# ---------------------------------------------------------------------------
# Instructions — verbatim from OpenRewriteEval (Shu et al. 2023)
# ---------------------------------------------------------------------------

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

REWRITE_TEMPLATE = """You will rewrite the text below according to the instruction.
Return ONLY the rewritten text, with no preamble or commentary.

Instruction: {instruction}

Text:
{text}

Rewritten text:"""


# ---------------------------------------------------------------------------
# Dataset loading (MuSiQue)
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
# Model loading + generation
# ---------------------------------------------------------------------------

def load_rewriter(model_id: str, use_4bit: bool = False):
    print(f"Loading rewriter: {model_id} (4-bit={use_4bit})", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    print(f"  device map: {getattr(model, 'hf_device_map', 'n/a')}", flush=True)
    return tok, model


@torch.no_grad()
def generate(
    tokenizer,
    model,
    user_prompt: str,
    *,
    temperature: float,
    max_new_tokens: int,
):
    messages = [{"role": "user", "content": user_prompt}]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = user_prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_chain(
    tokenizer,
    model,
    E0: str,
    instruction: str,
    *,
    n_iterations: int,
    temperature: float,
    max_new_tokens: int,
):
    chain = [E0]
    current = E0
    for _ in range(n_iterations):
        prompt = REWRITE_TEMPLATE.format(instruction=instruction, text=current)
        current = generate(
            tokenizer, model, prompt,
            temperature=temperature, max_new_tokens=max_new_tokens,
        )
        chain.append(current)
    return chain


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_done_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    missing = [k for k in CHAIN_KEYS if k not in df.columns]
    if missing:
        print(f"WARNING: existing CSV missing columns {missing}, starting fresh.", flush=True)
        return set()
    return {tuple(row[k] for k in CHAIN_KEYS) for _, row in df[CHAIN_KEYS].drop_duplicates().iterrows()}


def append_rows(csv_path: Path, rows: list):
    df = pd.DataFrame(rows)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate rewriting chains with repeated runs for P/A/U analysis."
    )
    parser.add_argument(
        "--model",
        default="allenai/OLMo-2-0325-32B-Instruct",
        help="HF model id of the rewriter (default: OLMo-2 32B Instruct).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to MuSiQue dev jsonl (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--n-per-hop", type=int, default=5,
        help="Questions per hop count (2/3/4). Default 5.",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=3,
        help="Number of rewriting steps (E0 -> E1 -> ... -> En). Default 3.",
    )
    parser.add_argument(
        "--n-repetitions", type=int, default=5,
        help="Number of stochastic repetitions per (qid, instruction, wording). Default 5.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature. Must be > 0 for P/A/U. Default 0.7.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048,
        help="Max new tokens per rewrite call. Default 2048.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for dataset balancing.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run only on the 2-hop pilot question, 1 repetition.",
    )
    parser.add_argument(
        "--only-supporting", action="store_true",
        help="Use only supporting paragraphs as E0. Default: all 20 paragraphs.",
    )
    parser.add_argument(
        "--use-4bit", action="store_true",
        help="Enable 4-bit NF4 quantization (for lisa/3090). Default: bfloat16.",
    )
    args = parser.parse_args()

    if args.temperature <= 0:
        print("ERROR: --temperature must be > 0 for P/A/U analysis.", file=sys.stderr)
        sys.exit(1)

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
        print("HF_TOKEN not set — proceeding without login (fine for public models)", flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\nLoading MuSiQue from {args.dataset}", flush=True)
    raw = load_musique(args.dataset)
    print(f"  loaded {len(raw)} items", flush=True)

    if args.smoke_test:
        questions = [it for it in raw if it["id"] == "2hop__635544_110949"]
        if not questions:
            print("ERROR: pilot question not found in dataset", file=sys.stderr)
            sys.exit(1)
        n_repetitions = args.n_repetitions
        print(f"\n*** SMOKE TEST: 1 question, {n_repetitions} repetitions ***", flush=True)
    else:
        questions = balance_by_hop(raw, args.n_per_hop, args.seed)
        n_repetitions = args.n_repetitions
        print(f"\nUsing {len(questions)} questions, balanced across hop counts", flush=True)

    done = load_done_keys(args.output)
    if done:
        print(f"\nResume: {len(done)} chains already done — will skip them.", flush=True)

    n_wordings = sum(len(pool) for pool in ALL_INSTRUCTIONS.values())
    total_chains = len(questions) * n_wordings * n_repetitions
    print(f"\nPlan: {len(questions)} questions x {n_wordings} wordings x {n_repetitions} repetitions = {total_chains} chains")
    print(f"      each chain = {args.n_iterations} steps + 1 baseline (E0) = {args.n_iterations + 1} rows")
    print(f"      total rows expected: {total_chains * (args.n_iterations + 1)}")

    tokenizer, model = load_rewriter(args.model, use_4bit=args.use_4bit)

    n_done = 0
    n_to_do = total_chains - len(done)
    t_start = time.time()

    for q in questions:
        qid = q["id"]
        question_text = q["question"]
        E0 = build_E0(q, only_supporting=args.only_supporting)

        for (group, instruction_type), pool in ALL_INSTRUCTIONS.items():
            for run, instruction in enumerate(pool):
                for rep in range(n_repetitions):
                    key = (qid, group, instruction_type, run, rep)
                    if key in done:
                        continue

                    t0 = time.time()
                    chain = run_chain(
                        tokenizer, model, E0, instruction,
                        n_iterations=args.n_iterations,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )
                    elapsed = time.time() - t0

                    rows = []
                    for step, text in enumerate(chain):
                        rows.append({
                            "qid": qid,
                            "question": question_text,
                            "group": group,
                            "instruction_type": instruction_type,
                            "run": run,
                            "repetition": rep,
                            "instruction_used": instruction if step > 0 else "",
                            "step": step,
                            "text": text,
                            "n_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                        })
                    append_rows(args.output, rows)
                    n_done += 1

                    avg = (time.time() - t_start) / max(n_done, 1)
                    remaining = (n_to_do - n_done) * avg
                    print(
                        f"[{n_done}/{n_to_do}] {qid} | {group}/{instruction_type}/run{run}/rep{rep} "
                        f"| {elapsed:.1f}s | ETA {remaining/60:.1f} min",
                        flush=True,
                    )

    print(f"\nDone. Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
