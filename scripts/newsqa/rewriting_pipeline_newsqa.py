"""
Rewriting pipeline for NewsQA (Maluuba) — generates rewriting chains E_0 -> E_1 -> E_2 -> E_3.

Dataset
-------
Reads combined-newsqa-data-v1.csv. Each row is one (story, question) pair with
columns: story_id, question, answer_char_ranges, is_answer_absent,
is_question_bad, validated_answers, story_text.

For each story we pick the *best-validated* answerable question and treat it
as the chain's gold Q&A.

Why one question per story
- The rewriting target is the article (E_0), not the question. Running multiple
  questions on the same E_0 chain would just multiply downstream eval cost
  without changing the rewriting itself.
- Per the NewsQA paper: 86% of questions have agreed answers (consensus or
  post-validation). We restrict to questions with at least one validated span
  agreed by >= 2 crowdworkers, dropping bad/null questions.

Why we keep ALL validated answer spans as gold (not just the top one)
- NewsQA's official EM/F1 takes the *max* over the set of human answers (like
  SQuAD/MuSiQue with aliases). The chain CSV stores the top span plus the
  alternatives in `gold_answer_aliases` (||-joined) so the F1 evaluator can
  reproduce the official metric.

Output schema
-------------
qid, question, gold_answer, gold_answer_aliases, group, instruction_type, run,
instruction_used, step, text, n_tokens

qid is the NewsQA story_id (e.g. "./cnn/stories/<sha1>.story").
text at step 0 is the raw story_text (E_0); at step>0 is the rewrite.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET_PATH = Path(os.environ.get(
    "NEWSQA_DATASET",
    "/mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv"
))
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "newsqa" / "rewriting_chains_newsqa.csv"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

ALIAS_SEP = "||"


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
# Dataset loading (NewsQA — CSV format)
# ---------------------------------------------------------------------------

def _span_to_text(span_key: str, story_text: str) -> str | None:
    """Convert a span key (e.g. "294:297" or "10:20,30:40") to the answer text.

    Multi-span keys are joined with a single space (post-cleanup style of the
    NewsQA paper §3.5). Returns None if any sub-span fails to parse.
    """
    parts: list[str] = []
    try:
        for span in span_key.split(","):
            s_str, e_str = span.split(":")
            s, e = int(s_str), int(e_str)
            parts.append(story_text[s:e])
    except (ValueError, IndexError):
        return None
    out = " ".join(p.strip() for p in parts if p.strip()).strip()
    return out or None


def parse_validated_answers(value, story_text: str) -> tuple[list[tuple[str, int]], int, int]:
    """Parse the validated_answers JSON cell.

    NewsQA format: {"none": k0, "294:297": k1, "10:20,30:40": k2, "badQuestion": k3, ...}
    Keys can be:
      - "none"          -> the validators said there's no answer
      - "badQuestion"   -> the validators rejected the question
      - "<s>:<e>"       -> a single span
      - "<s>:<e>,<s>:<e>"  -> a multi-span answer

    Returns
    -------
    (answer_strings_with_counts, n_none, n_bad)
    """
    if not isinstance(value, str) or not value.strip():
        return [], 0, 0
    try:
        d = json.loads(value)
    except json.JSONDecodeError:
        return [], 0, 0

    answers: list[tuple[str, int]] = []
    n_none = 0
    n_bad = 0
    for key, count in d.items():
        if key == "none":
            n_none = int(count)
            continue
        if key == "badQuestion":
            n_bad = int(count)
            continue
        ans = _span_to_text(key, story_text)
        if ans:
            answers.append((ans, int(count)))
    answers.sort(key=lambda t: -t[1])
    return answers, n_none, n_bad


def parse_sourcer_agreement(value, story_text: str) -> tuple[list[tuple[str, int]], int]:
    """Parse the answer_char_ranges field for initial-stage agreement.

    Per the NewsQA paper §3.4, the sourcing stage asks multiple Answerers per
    question (avg 2.73). When >=2 Answerers select the SAME span, the question
    is considered already agreed-upon and skips the validation step (which is
    why only 43.2% of questions even reach validation).

    Format: '294:297|None|10:20,30:40' — one answer per Answerer, separated by
    '|'. 'None' means that Answerer said the question is unanswerable.

    Returns (spans_with_counts, n_none).
    """
    if not isinstance(value, str) or not value.strip():
        return [], 0
    counts: dict[str, int] = {}
    n_none = 0
    for raw in value.split("|"):
        raw = raw.strip()
        if not raw or raw == "None":
            n_none += 1 if raw == "None" else 0
            continue
        ans = _span_to_text(raw, story_text)
        if ans:
            counts[ans] = counts.get(ans, 0) + 1
    answers = sorted(counts.items(), key=lambda t: -t[1])
    return answers, n_none


def pick_best_question_for_story(rows: list[dict]) -> dict | None:
    """Given all CSV rows for one story_id, return a record for the best question.

    A question is acceptable if EITHER:
      (A) the validation step ran and at least one span got count >= 2, OR
      (B) the sourcing step already had >= 2 Answerers agree on the same span
          (these questions skip validation by design — paper §3.4).

    In both cases we also require:
      - is_answer_absent < 0.5
      - is_question_bad  < 0.5
      - the agreed span(s) outvote 'None'/'badQuestion'

    Tie-break: more aliases first, then longer question text.

    Returns dict {id, text, question, answer, aliases} or None.
    """
    candidates = []
    for r in rows:
        try:
            if float(r.get("is_answer_absent") or 0) >= 0.5:
                continue
            if float(r.get("is_question_bad") or 0) >= 0.5:
                continue
        except (TypeError, ValueError):
            continue
        story_text = r.get("story_text") or ""
        if not story_text:
            continue

        # (A) validated_answers — present only for the ~43% of Q that needed validation
        v_spans, v_none, v_bad = parse_validated_answers(r.get("validated_answers"), story_text)
        v_agreed = [(ans, c) for ans, c in v_spans if c >= 2]
        if v_agreed and v_agreed[0][1] > max(v_none, v_bad):
            aliases = sorted({ans for ans, _ in v_agreed})
            candidates.append({
                "id": r["story_id"],
                "text": story_text,
                "question": r["question"],
                "answer": v_agreed[0][0],
                "aliases": aliases,
            })
            continue

        # (B) initial sourcer agreement — for Q that skipped validation
        s_spans, s_none = parse_sourcer_agreement(r.get("answer_char_ranges"), story_text)
        s_agreed = [(ans, c) for ans, c in s_spans if c >= 2]
        if s_agreed and s_agreed[0][1] > s_none:
            aliases = sorted({ans for ans, _ in s_agreed})
            candidates.append({
                "id": r["story_id"],
                "text": story_text,
                "question": r["question"],
                "answer": s_agreed[0][0],
                "aliases": aliases,
            })

    if not candidates:
        return None
    candidates.sort(key=lambda c: (-len(c["aliases"]), -len(c["question"])))
    return candidates[0]


def load_newsqa(path: Path) -> list[dict]:
    """Load NewsQA CSV and return one usable record per story_id.

    The CSV is large (~450 MB / ~120k rows) so we read it once with pandas and
    then group rows by story_id in memory.
    """
    print(f"  reading CSV (this may take a minute)...", flush=True)
    df = pd.read_csv(
        path,
        dtype={
            "story_id": "string",
            "question": "string",
            "answer_char_ranges": "string",
            "validated_answers": "string",
            "story_text": "string",
        },
        keep_default_na=False,
    )
    print(f"  CSV loaded: {len(df):,} rows · {df['story_id'].nunique():,} stories", flush=True)

    items: list[dict] = []
    for _, story_rows in df.groupby("story_id", sort=False):
        rec = pick_best_question_for_story(story_rows.to_dict(orient="records"))
        if rec is not None:
            items.append(rec)
    print(f"  answerable stories after quality filter: {len(items):,}", flush=True)
    return items


def sample_items(items: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = list(items)
    rng.shuffle(pool)
    return pool[:n]


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
def generate(tokenizer, model, user_prompt: str, *, temperature: float, max_new_tokens: int):
    messages = [{"role": "user", "content": user_prompt}]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = user_prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_chain(tokenizer, model, E0: str, instruction: str, *, n_iterations: int,
              temperature: float, max_new_tokens: int):
    chain = [E0]
    current = E0
    for _ in range(n_iterations):
        prompt = REWRITE_TEMPLATE.format(instruction=instruction, text=current)
        current = generate(tokenizer, model, prompt, temperature=temperature,
                           max_new_tokens=max_new_tokens)
        chain.append(current)
    return chain


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
    parser = argparse.ArgumentParser(description="Generate NewsQA rewriting chains on GPU.")
    parser.add_argument("--model", default="allenai/OLMo-2-0325-32B-Instruct",
                        help="HF model id of the rewriter.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH,
                        help=f"Path to combined-newsqa-data-v1.json (default: {DEFAULT_DATASET_PATH}).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).")
    parser.add_argument("--n-questions", type=int, default=300,
                        help="Number of stories to sample (default: 300).")
    parser.add_argument("--n-iterations", type=int, default=3,
                        help="Number of rewriting steps (E0 -> E1 -> ... -> En). Default 3.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for the rewriter. Default 0.7.")
    # NewsQA articles average 30+ sentences / much longer than MuSiQue paragraphs,
    # so we need more headroom. Even shrink instructions still produce ~1-2k tokens.
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max new tokens per rewrite call. Default 4096 (NewsQA articles are long).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for story sampling.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run only on the first story (1 chain, smoke test).")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Enable 4-bit NF4 quantization. Default: bfloat16.")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("HF login OK", flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\nLoading NewsQA from {args.dataset}", flush=True)
    all_items = load_newsqa(args.dataset)

    if args.smoke_test:
        questions = all_items[:1]
        print(f"\n*** SMOKE TEST: 1 story ***", flush=True)
    else:
        questions = sample_items(all_items, args.n_questions, args.seed)
        print(f"\nUsing {len(questions)} stories (sampled)", flush=True)

    done = load_done_keys(args.output)
    if done:
        print(f"\nResume: {len(done)} chains already in {args.output} — will skip them.", flush=True)

    total_chains = len(questions) * sum(len(pool) for pool in ALL_INSTRUCTIONS.values())
    print(f"\nPlan: {len(questions)} stories x 4 instructions x 3 wordings = {total_chains} chains")
    print(f"      each chain = {args.n_iterations} steps + 1 baseline (E0) = {args.n_iterations+1} rows")
    print(f"      total rows expected: {total_chains * (args.n_iterations + 1)}")

    tokenizer, model = load_rewriter(args.model, use_4bit=args.use_4bit)

    n_done = 0
    n_to_do = total_chains - len(done)
    t_start = time.time()

    for q in questions:
        qid = q["id"]
        question_text = q["question"]
        gold_answer = q["answer"]
        aliases_str = ALIAS_SEP.join(q["aliases"])
        E0 = q["text"]

        for (group, instruction_type), pool in ALL_INSTRUCTIONS.items():
            for run, instruction in enumerate(pool):
                key = (qid, group, instruction_type, run)
                if key in done:
                    continue

                t0 = time.time()
                chain = run_chain(tokenizer, model, E0, instruction,
                                  n_iterations=args.n_iterations,
                                  temperature=args.temperature,
                                  max_new_tokens=args.max_new_tokens)
                elapsed = time.time() - t0

                rows = []
                for step, text in enumerate(chain):
                    rows.append({
                        "qid": qid,
                        "question": question_text,
                        "gold_answer": gold_answer,
                        "gold_answer_aliases": aliases_str,
                        "group": group,
                        "instruction_type": instruction_type,
                        "run": run,
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
                    f"[{n_done}/{n_to_do}] {qid} | {group}/{instruction_type}/run{run} "
                    f"| {elapsed:.1f}s | ETA {remaining/60:.1f} min",
                    flush=True,
                )

    print(f"\nDone. Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
