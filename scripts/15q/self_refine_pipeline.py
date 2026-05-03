"""
Self-Refine pipeline (RQ3) — generates the rewriting chains E_0 -> Ẽ_1 -> Ẽ_2 -> Ẽ_3
using a three-stage Rewriter / Critic / Refiner loop.

Pipeline (per question, per (group, instruction_type), per wording = run)
-------------------------------------------------------------------------
At each step t:
  1. Rewriter — receives Ẽ_{t-1} and the instruction → produces a draft E_draft.
  2. Critic   — receives the original text E_0 and E_draft → identifies facts
                that were changed, removed, or distorted. Feedback must be
                specific (quote the problematic part) and actionable (state
                the correct version).
  3. Refiner  — receives E_draft and the feedback → produces the corrected
                text Ẽ_t, preserving the requested style but fixing factual
                errors. Ẽ_t becomes the input of step t+1.

Metrics downstream are computed on Ẽ_t. We also persist the intermediate
draft and the critic feedback so the chain is fully auditable.

Output schema (extends rewriting_chains*.csv)
---------------------------------------------
Same key columns as the baseline pipeline (qid, group, instruction_type,
run, step, text, n_tokens, ...) plus:
  - draft_text       : E_draft produced by the rewriter (empty for step=0)
  - critic_feedback  : critic output (empty for step=0)
  - draft_n_tokens   : token count of E_draft

Why this script (vs rewriting_pipeline.py)
------------------------------------------
- Same model is used for all three roles (Rewriter, Critic, Refiner).
- Resume support, smoke-test mode, CLI flags mirror the baseline pipeline
  to keep the experimental setup comparable.
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
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "15q" / "self_refine_chains_15q.csv"
DEFAULT_BASELINE_CSV = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q.csv"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]


# ---------------------------------------------------------------------------
# Instructions — verbatim from OpenRewriteEval (Shu et al. 2023)
# (kept identical to rewriting_pipeline.py so RQ1/RQ3 are directly comparable)
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


# ---------------------------------------------------------------------------
# Prompt templates for the three roles
# ---------------------------------------------------------------------------

REWRITER_TEMPLATE = """You will rewrite the text below according to the instruction.
The text consists of multiple paragraphs. You MUST rewrite ALL of them — do not skip, omit, or merge any paragraph.
Return ONLY the rewritten text, preserving the same multi-paragraph structure, with no preamble or commentary.

Instruction: {instruction}

Text:
{text}

Rewritten text (all paragraphs):"""

CRITIC_TEMPLATE = """You are a fact-checking critic. Compare the DRAFT to the ORIGINAL and review every factual aspect: facts that were changed, removed, or distorted (including dates, numbers, names, relations, and entities).

Rules:
- Always produce a critique. The critique must include a Verdict line and an Issues block, even when the draft is faithful.
- Be specific: quote the exact problematic span from the DRAFT in double quotes.
- Be actionable: for each issue, state the correct version grounded in the ORIGINAL.
- Ignore stylistic differences (tone, formality, wording) that do not change facts.

Output format (always emit both sections):
Verdict: <one short sentence summarizing factual fidelity of the draft against the original>
Issues:
- Issue: "<quoted span from draft>" — Correction: <correct version from the original>
- ...
(If no factual issues are present, write a single line under Issues: "- None.")

ORIGINAL:
{original}
{prior_feedback_block}
DRAFT:
{draft}

Feedback:"""

PRIOR_FEEDBACK_BLOCK_TEMPLATE = """
PRIOR FEEDBACK (from earlier iterations on previous drafts of this text — do not repeat these mistakes):
{prior_feedback}
"""

REFINER_TEMPLATE = """You will revise the DRAFT using the FEEDBACK so that every factual error is corrected, while keeping the style of the draft (tone, formality, length, register) as close as possible to what it already is.

Rules:
- Apply every correction listed under "Issues" in the feedback.
- You MUST reproduce ALL paragraphs of the draft — do not skip, omit, or merge any paragraph.
- Do not introduce new facts that are not in the feedback or the draft.
- Return ONLY the corrected text, preserving the same multi-paragraph structure, with no preamble or commentary.

DRAFT:
{draft}

FEEDBACK:
{feedback}
{prior_feedback_block}
Corrected text (all paragraphs):"""

REFINER_PRIOR_FEEDBACK_BLOCK_TEMPLATE = """
PRIOR FEEDBACK (from earlier iterations — these issues should already be fixed; do not reintroduce them):
{prior_feedback}
"""


# ---------------------------------------------------------------------------
# Dataset loading (MuSiQue) — identical to rewriting_pipeline.py
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

def load_model(model_id: str, use_4bit: bool = False):
    print(f"Loading model: {model_id} (4-bit={use_4bit})", flush=True)
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


# ---------------------------------------------------------------------------
# Self-refine step: Rewriter → Critic → Refiner
# ---------------------------------------------------------------------------

def _format_prior_feedback(prior_feedbacks: list, template: str) -> str:
    """Render past feedbacks as a numbered block, or empty string if none.

    Madaan et al. 2023 (Self-Refine) keep the history of past experiences by
    appending previous outputs to the prompt so the model can avoid repeating
    mistakes. We carry only feedbacks (not drafts) — they are the dense signal
    for our factuality task and keep prompt growth bounded across iterations.
    """
    if not prior_feedbacks:
        return ""
    rendered = "\n".join(
        f"[Iteration {i+1}]\n{fb}" for i, fb in enumerate(prior_feedbacks)
    )
    return template.format(prior_feedback=rendered)


def self_refine_step(
    tokenizer,
    model,
    *,
    E0: str,
    prev_text: str,
    instruction: str,
    prior_feedbacks: list,
    rewriter_temperature: float,
    critic_temperature: float,
    refiner_temperature: float,
    rewriter_max_new_tokens: int,
    critic_max_new_tokens: int,
    refiner_max_new_tokens: int,
):
    """One self-refine iteration. Returns (E_draft, feedback, E_tilde)."""
    # 1. Rewriter
    rewriter_prompt = REWRITER_TEMPLATE.format(instruction=instruction, text=prev_text)
    E_draft = generate(
        tokenizer, model, rewriter_prompt,
        temperature=rewriter_temperature,
        max_new_tokens=rewriter_max_new_tokens,
    )

    # 2. Critic — compares E_draft against the ORIGINAL E0 (not the previous step)
    #    so factual drift across iterations is caught and not silently propagated.
    #    Past feedbacks are appended (Madaan 2023, "retain history of past experiences").
    critic_prior = _format_prior_feedback(prior_feedbacks, PRIOR_FEEDBACK_BLOCK_TEMPLATE)
    critic_prompt = CRITIC_TEMPLATE.format(
        original=E0, draft=E_draft, prior_feedback_block=critic_prior
    )
    feedback = generate(
        tokenizer, model, critic_prompt,
        temperature=critic_temperature,
        max_new_tokens=critic_max_new_tokens,
    )

    # 3. Refiner — always run, per §5.4 (Ẽₜ is always produced by the refiner).
    refiner_prior = _format_prior_feedback(
        prior_feedbacks, REFINER_PRIOR_FEEDBACK_BLOCK_TEMPLATE
    )
    refiner_prompt = REFINER_TEMPLATE.format(
        draft=E_draft, feedback=feedback, prior_feedback_block=refiner_prior
    )
    E_tilde = generate(
        tokenizer, model, refiner_prompt,
        temperature=refiner_temperature,
        max_new_tokens=refiner_max_new_tokens,
    )

    return E_draft, feedback, E_tilde


def run_chain(
    tokenizer,
    model,
    E0: str,
    instruction: str,
    *,
    n_iterations: int,
    rewriter_temperature: float,
    critic_temperature: float,
    refiner_temperature: float,
    rewriter_max_new_tokens: int,
    critic_max_new_tokens: int,
    refiner_max_new_tokens: int,
):
    """Iteratively self-refine E0. Returns a list of dicts, one per step.

    step=0 is the baseline (E0 only, no draft/feedback).
    step>=1 carries (E_draft, feedback, E_tilde) for the t-th iteration.
    """
    steps = [{"E_draft": "", "feedback": "", "E_tilde": E0}]
    current = E0
    prior_feedbacks: list = []
    for _ in range(n_iterations):
        E_draft, feedback, E_tilde = self_refine_step(
            tokenizer, model,
            E0=E0,
            prev_text=current,
            instruction=instruction,
            prior_feedbacks=prior_feedbacks,
            rewriter_temperature=rewriter_temperature,
            critic_temperature=critic_temperature,
            refiner_temperature=refiner_temperature,
            rewriter_max_new_tokens=rewriter_max_new_tokens,
            critic_max_new_tokens=critic_max_new_tokens,
            refiner_max_new_tokens=refiner_max_new_tokens,
        )
        steps.append({"E_draft": E_draft, "feedback": feedback, "E_tilde": E_tilde})
        current = E_tilde
        prior_feedbacks = prior_feedbacks + [feedback]
    return steps


# ---------------------------------------------------------------------------
# Load E0 texts from the baseline rewriting CSV
# ---------------------------------------------------------------------------

def load_e0_from_baseline(csv_path: Path) -> dict:
    """Return {qid: (question_text, E0_text)} from step=0 rows of the baseline CSV.

    Using the baseline CSV guarantees that E0 is byte-for-byte identical to what
    the rewriting pipeline used — same tokenizer, same paragraph order, same text.
    """
    df = pd.read_csv(csv_path)
    step0 = df[df["step"] == 0].drop_duplicates(subset=["qid"])
    result = {}
    for _, row in step0.iterrows():
        result[row["qid"]] = (row["question"], row["text"])
    return result


# ---------------------------------------------------------------------------
# Resume support
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
    parser = argparse.ArgumentParser(description="Generate self-refine chains on GPU (RQ3).")
    parser.add_argument(
        "--model",
        default="allenai/OLMo-3.1-32B-Instruct",
        help="HF model id used for all three roles. Must match the baseline rewriting pipeline model.",
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
        help="Number of self-refine steps (E0 -> Ẽ1 -> ... -> Ẽn). Default 3.",
    )
    parser.add_argument(
        "--rewriter-temperature", type=float, default=0.7,
        help="Sampling temperature for the rewriter. Default 0.7 — matches baseline pipeline. "
             "Do NOT use 0.0: greedy decoding on the full 20-paragraph E0 causes the model to "
             "summarize only the most salient topic instead of rewriting all paragraphs.",
    )
    parser.add_argument(
        "--critic-temperature", type=float, default=0.0,
        help="Sampling temperature for the critic. Default 0.0 (deterministic fact-checking).",
    )
    parser.add_argument(
        "--refiner-temperature", type=float, default=0.3,
        help="Sampling temperature for the refiner. Default 0.3.",
    )
    parser.add_argument(
        "--rewriter-max-new-tokens", type=int, default=8192,
        help="Max new tokens for the rewriter. Default 8192 (E0 ~2300 tokens, verbose rewrites can exceed 4096).",
    )
    parser.add_argument(
        "--critic-max-new-tokens", type=int, default=1024,
        help="Max new tokens for the critic. Default 1024.",
    )
    parser.add_argument(
        "--refiner-max-new-tokens", type=int, default=8192,
        help="Max new tokens for the refiner. Default 8192 (matches rewriter).",
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
        "--n-runs", type=int, default=None,
        help="Limit to the first N wordings per instruction type (1-3). "
             "Use --n-runs 1 with --smoke-test --n-iterations 1 for a quick sanity check.",
    )
    parser.add_argument(
        "--qids-file", type=Path, default=None,
        help="Path to a text file with one qid per line. If given, only those questions are used.",
    )
    parser.add_argument(
        "--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV,
        help=f"Path to the baseline rewriting CSV (step=0 rows used as E0). "
             f"Default: {DEFAULT_BASELINE_CSV}. "
             f"When provided, E0 is taken directly from this file instead of rebuilding from MuSiQue, "
             f"guaranteeing identical starting text.",
    )
    parser.add_argument(
        "--only-supporting", action="store_true",
        help="Use only supporting paragraphs as E0 (only applies when --baseline-csv is not used).",
    )
    parser.add_argument(
        "--use-4bit", action="store_true",
        help="Enable 4-bit NF4 quantization (for lisa/3090). Default: bfloat16.",
    )
    args = parser.parse_args()

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

    # Load E0 from the baseline CSV (guaranteed identical starting text)
    if not args.baseline_csv.exists():
        print(f"ERROR: baseline CSV not found: {args.baseline_csv}", file=sys.stderr)
        sys.exit(1)
    e0_lookup = load_e0_from_baseline(args.baseline_csv)
    print(f"\nLoaded E0 for {len(e0_lookup)} questions from {args.baseline_csv}", flush=True)

    # Determine which qids to process
    if args.smoke_test:
        qids_to_run = ["2hop__635544_110949"]
        print(f"\n*** SMOKE TEST: 1 question (pilot 2-hop) ***", flush=True)
    elif args.qids_file:
        qids_to_run = [q.strip() for q in args.qids_file.read_text().splitlines() if q.strip()]
        print(f"\nUsing {len(qids_to_run)} questions from {args.qids_file}", flush=True)
    else:
        # Fall back to MuSiQue for full dataset runs
        if not args.dataset.exists():
            print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
            sys.exit(1)
        raw = load_musique(args.dataset)
        qids_to_run = [it["id"] for it in balance_by_hop(raw, args.n_per_hop, args.seed)]
        print(f"\nUsing {len(qids_to_run)} questions, balanced across hop counts", flush=True)

    # Validate all qids have E0 in the baseline CSV
    missing = [qid for qid in qids_to_run if qid not in e0_lookup]
    if missing:
        print(f"ERROR: {len(missing)} qids not found in baseline CSV: {missing}", file=sys.stderr)
        sys.exit(1)

    done = load_done_keys(args.output)
    if done:
        print(f"\nResume: {len(done)} chains already in {args.output} — will skip them.", flush=True)

    n_runs_effective = args.n_runs if args.n_runs is not None else max(len(p) for p in ALL_INSTRUCTIONS.values())
    total_chains = len(qids_to_run) * sum(min(len(pool), n_runs_effective) for pool in ALL_INSTRUCTIONS.values())
    print(f"\nPlan: {len(qids_to_run)} questions x 4 instructions x {n_runs_effective} wording(s) = {total_chains} chains")
    print(f"      each chain = {args.n_iterations} steps + 1 baseline (E0) = {args.n_iterations+1} rows")
    print(f"      total rows expected: {total_chains * (args.n_iterations + 1)}")

    tokenizer, model = load_model(args.model, use_4bit=args.use_4bit)

    n_done = 0
    n_to_do = total_chains - len(done)
    t_start = time.time()

    for qid in qids_to_run:
        question_text, E0 = e0_lookup[qid]

        for (group, instruction_type), pool in ALL_INSTRUCTIONS.items():
            for run, instruction in enumerate(pool[: args.n_runs]):
                key = (qid, group, instruction_type, run)
                if key in done:
                    continue

                t0 = time.time()
                steps = run_chain(
                    tokenizer, model, E0, instruction,
                    n_iterations=args.n_iterations,
                    rewriter_temperature=args.rewriter_temperature,
                    critic_temperature=args.critic_temperature,
                    refiner_temperature=args.refiner_temperature,
                    rewriter_max_new_tokens=args.rewriter_max_new_tokens,
                    critic_max_new_tokens=args.critic_max_new_tokens,
                    refiner_max_new_tokens=args.refiner_max_new_tokens,
                )
                elapsed = time.time() - t0

                rows = []
                for step, s in enumerate(steps):
                    text = s["E_tilde"]
                    draft = s["E_draft"]
                    rows.append({
                        "qid": qid,
                        "question": question_text,
                        "group": group,
                        "instruction_type": instruction_type,
                        "run": run,
                        "instruction_used": instruction if step > 0 else "",
                        "step": step,
                        "text": text,
                        "draft_text": draft,
                        "critic_feedback": s["feedback"],
                        "n_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                        "draft_n_tokens": (
                            len(tokenizer.encode(draft, add_special_tokens=False))
                            if draft else 0
                        ),
                    })
                append_rows(args.output, rows)
                n_done += 1

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
