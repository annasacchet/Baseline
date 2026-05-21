"""
Rewriting pipeline — generates the rewriting chains E_0 -> E_1 -> E_2 -> E_3
on the GPU server (Homer @ MITEL Lab).

Pipeline
--------
For each MuSiQue question (filtered/sampled), and for each (group, instruction_type):
  - take the 3 wordings of the instruction (OpenRewriteEval, Shu et al. 2023)
  - one wording = one run (run 0, run 1, run 2)
  - apply the rewriter iteratively for n_iterations steps
  - save one row per (qid, group, instruction_type, run, step) to CSV

Output schema matches the existing rewriting_chains*.csv files in results/.

Why this script (vs the Colab notebook)
---------------------------------------
- No google.colab dependencies (no files.upload, no userdata)
- HF_TOKEN read from environment variable
- Resume support via CSV (skip already-done chains)
- CLI flags for smoke test, model override, n questions
- Designed to run inside `tmux` on Homer
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

REWRITE_TEMPLATE = """You are a precise text rewriting assistant. Your task is to rewrite the text provided inside the XML tags according to the specific instruction.

<source_text>
{text}
</source_text>

Instruction: {instruction}

Strict Rule: Return ONLY the rewritten text. Do not include any preamble, introduction, markdown formatting outside the text, or commentary.

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
    """Concatenate paragraphs (titles + texts) into the source text E0.

    only_supporting=True  → just the gold supporting paragraphs (clean)
    only_supporting=False → all 20 paragraphs incl. distractors (matches the pilot)
    """
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
    """Single-prompt generation via the model's chat template."""
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
    """Iteratively rewrite E0 with the same instruction. Returns [E0, E1, ..., En]."""
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
    """Read the existing CSV (if any) and return the set of (qid,group,instruction_type,run) already done."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return {tuple(row[k] for k in CHAIN_KEYS) for _, row in df[CHAIN_KEYS].drop_duplicates().iterrows()}


def append_rows(csv_path: Path, rows: list):
    """Append rows to the CSV, creating it with header if it doesn't exist."""
    df = pd.DataFrame(rows)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate rewriting chains on GPU.")
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
        "--n-per-hop", type=int, default=405,
        help="Questions per hop count (2/3/4). Default 405 = full balanced subset.",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=3,
        help="Number of rewriting steps (E0 -> E1 -> ... -> En). Default 3.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for the rewriter. Default 0.7.",
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
        help="Run only on the 2-hop pilot question (matches the existing pilot CSV).",
    )
    parser.add_argument(
        "--qids-file", type=Path, default=None,
        help="Path to a file with one MuSiQue qid per line. When set, run on "
             "exactly these qids (overrides --n-per-hop and --smoke-test). "
             "Used to pin the 4-bit run to the same 15 qids as the full-precision "
             "pilot for an apples-to-apples quantization comparison.",
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

    # Sanity checks
    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # HuggingFace login (optional — only needed for gated models)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("HF login OK", flush=True)
    else:
        print("HF_TOKEN not set — proceeding without login (fine for public models)", flush=True)

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load + select questions
    print(f"\nLoading MuSiQue from {args.dataset}", flush=True)
    raw = load_musique(args.dataset)
    print(f"  loaded {len(raw)} items", flush=True)
    print(f"  hop distribution: {dict((h, sum(1 for it in raw if hop_count(it)==h)) for h in (2,3,4))}", flush=True)

    if args.qids_file:
        # Run on the exact qids listed in the file (one per line, empty/# lines skipped).
        wanted = set()
        with open(args.qids_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    wanted.add(line)
        questions = [it for it in raw if it["id"] in wanted]
        missing = wanted - {q["id"] for q in questions}
        if missing:
            print(f"WARNING: {len(missing)} qids from {args.qids_file} not found in dataset:", file=sys.stderr)
            for m in sorted(missing):
                print(f"  {m}", file=sys.stderr)
        if not questions:
            print(f"ERROR: no qids from {args.qids_file} matched the dataset", file=sys.stderr)
            sys.exit(1)
        print(f"\n*** PINNED to {len(questions)}/{len(wanted)} qids from {args.qids_file} ***", flush=True)
    elif args.smoke_test:
        # Pin to the same pilot question used for the existing CSV
        questions = [it for it in raw if it["id"] == "2hop__635544_110949"]
        if not questions:
            print("ERROR: pilot question not found in dataset", file=sys.stderr)
            sys.exit(1)
        print(f"\n*** SMOKE TEST: 1 question (pilot 2-hop) ***", flush=True)
    else:
        questions = balance_by_hop(raw, args.n_per_hop, args.seed)
        print(f"\nUsing {len(questions)} questions, balanced across hop counts", flush=True)

    # Resume support
    done = load_done_keys(args.output)
    if done:
        print(f"\nResume: {len(done)} chains already in {args.output} — will skip them.", flush=True)

    # Total chains to do
    total_chains = (
        len(questions)
        * sum(len(pool) for pool in ALL_INSTRUCTIONS.values())  # = 4 instructions x 3 wordings = 12
    )
    print(f"\nPlan: {len(questions)} questions x 4 instructions x 3 wordings = {total_chains} chains")
    print(f"      each chain = {args.n_iterations} steps + 1 baseline (E0) = {args.n_iterations+1} rows")
    print(f"      total rows expected: {total_chains * (args.n_iterations + 1)}")

    # Load the model
    tokenizer, model = load_rewriter(args.model, use_4bit=args.use_4bit)

    # Main loop
    n_done = 0
    n_to_do = total_chains - len(done)
    t_start = time.time()

    for q in questions:
        qid = q["id"]
        question_text = q["question"]
        E0 = build_E0(q, only_supporting=args.only_supporting)

        for (group, instruction_type), pool in ALL_INSTRUCTIONS.items():
            for run, instruction in enumerate(pool):
                key = (qid, group, instruction_type, run)
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
                        "instruction_used": instruction if step > 0 else "",
                        "step": step,
                        "text": text,
                        # Token count using the rewriter's own tokenizer.
                        # Skip special tokens so the value reflects the actual
                        # generation load, not the chat-template overhead.
                        "n_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                    })
                append_rows(args.output, rows)
                n_done += 1

                # Progress
                avg = (time.time() - t_start) / max(n_done, 1)
                remaining = (n_to_do - n_done) * avg
                print(
                    f"[{n_done}/{n_to_do}] {qid} | {group}/{instruction_type}/run{run} "
                    f"| {elapsed:.1f}s | ETA {remaining/60:.1f} min",
                    flush=True,
                )

    print(f"\nDone. Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
