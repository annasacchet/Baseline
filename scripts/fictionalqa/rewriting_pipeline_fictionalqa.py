"""
Rewriting pipeline for FictionalQA — generates rewriting chains E_0 -> E_1 -> E_2 -> E_3.

Dataset
-------
Reads jwkirchenbauer/fictionalqa from the Hugging Face Hub. Two parquet files
are pulled:
  - fictions/        : the documents (5 webtext styles × 300 events = 1500 docs)
  - joined_qa/       : Q&A pairs joined with their fiction document, plus the
                       blind/informed grading from the paper's Q&A annotation
                       step.

For each fiction document we pick the *best* answerable question:
  - grade_blind == 0   → the LLM cannot answer without the document (the paper
                        considers only these "infeasible" questions in §4.4)
  - grade_informed == 1 → the LLM CAN answer when given the document
  - duplicate_relationship in {None, 'unique'} OR the row IS the duplicate root
                        → avoid duplicate questions across fiction styles
  - tie-break: shortest natural_answer first (more specific factoid)

This means the chain's gold Q&A genuinely tests whether the rewriting preserved
the fact — there's no chance of the QA model answering from prior knowledge
because the fact is fictional.

Output schema (matches the NewsQA pipeline)
-------------------------------------------
qid, question, gold_answer, gold_answer_aliases, group, instruction_type, run,
instruction_used, step, text, n_tokens

qid is the FictionalQA fiction_id (e.g. "event_000_style_blog_num_000").
text at step 0 is the raw fiction document; at step>0 it is the rewrite.
gold_answer = natural_answer; gold_answer_aliases also includes span_answer.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "fictionalqa" / "rewriting_chains_fictionalqa.csv"
HF_REPO = "jwkirchenbauer/fictionalqa"

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
# Dataset loading (FictionalQA — via HF Hub)
# ---------------------------------------------------------------------------

def download_parquet(filename: str) -> Path:
    """Download a single parquet file from jwkirchenbauer/fictionalqa to the HF cache."""
    return Path(hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=filename))


def load_fictionalqa(allowed_styles: list[str] | None = None) -> list[dict]:
    """Load FictionalQA and return one usable record per fiction document.

    Returns a list of dicts: {id, text, question, answer, aliases, style, event_id}.
    Each dict represents one chain (one fiction document with its best Q&A).
    """
    print("  downloading FictionalQA from HF ...", flush=True)
    qa_path = download_parquet("joined_qa/train-00000-of-00001.parquet")
    fic_path = download_parquet("fictions/train-00000-of-00001.parquet")

    df_qa = pd.read_parquet(qa_path)
    df_fic = pd.read_parquet(fic_path)
    print(f"  joined_qa: {len(df_qa):,} rows · fictions: {len(df_fic):,} docs", flush=True)

    if allowed_styles:
        df_fic = df_fic[df_fic["style"].isin(allowed_styles)]
        df_qa = df_qa[df_qa["style"].isin(allowed_styles)]
        print(f"  filtered to styles {allowed_styles}: {len(df_fic):,} fictions · {len(df_qa):,} QAs", flush=True)

    # Filter to genuinely-infeasible-blind, answerable-when-informed questions.
    quality = df_qa[(df_qa["grade_blind"] == 0) & (df_qa["grade_informed"] == 1)].copy()

    # Drop near-duplicate questions across fiction styles for the same event.
    # If duplicate_root != fiction_id+question_id, this row is a duplicate — skip
    # it and keep only the canonical root copy.
    quality["self_id"] = quality["question_id"]
    is_root = (quality["duplicate_root"].isna()) | (quality["duplicate_root"] == quality["self_id"])
    is_unique = quality["duplicate_relationship"].isin([None, "", "unique"]) | quality["duplicate_relationship"].isna()
    quality = quality[is_root | is_unique].copy()
    print(f"  quality-filtered Qs: {len(quality):,}", flush=True)

    items: list[dict] = []
    for fid, group in quality.groupby("fiction_id", sort=False):
        # Pick the question with the shortest natural_answer (most specific fact).
        group = group.assign(_alen=group["natural_answer"].str.len())
        best = group.sort_values("_alen", na_position="last").iloc[0]
        nat = (best["natural_answer"] or "").strip()
        span = (best["span_answer"] or "").strip()
        if not nat:
            continue
        aliases = [nat]
        if span and span != nat:
            aliases.append(span)
        items.append({
            "id": fid,
            "event_id": best["event_id"],
            "style": best["style"],
            "text": best["fiction"],
            "question": best["question"],
            "answer": nat,
            "aliases": aliases,
        })
    print(f"  usable fictions (one Q each): {len(items):,}", flush=True)
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
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        current = generate(tokenizer, model, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
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
    parser = argparse.ArgumentParser(description="Generate FictionalQA rewriting chains on GPU.")
    parser.add_argument("--model", default="allenai/OLMo-3.1-32B-Instruct",
                        help="HF model id of the rewriter.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).")
    parser.add_argument("--n-fictions", type=int, default=300,
                        help="Number of fiction documents to sample (default: 300).")
    parser.add_argument("--styles", nargs="+", default=None,
                        choices=["news", "blog", "social", "corporate", "encyclopedia"],
                        help="Restrict to specific fiction styles. Default: all 5.")
    parser.add_argument("--n-iterations", type=int, default=3,
                        help="Number of rewriting steps (E0 -> E1 -> ... -> En). Default 3.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for the rewriter. Default 0.7.")
    # FictionalQA documents are similar in length to NewsQA (a few thousand
    # chars on average) so we keep the same headroom.
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max new tokens per rewrite call. Default 4096.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run only on the first fiction (1 chain).")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Enable 4-bit NF4 quantization. Default: bfloat16.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("HF login OK", flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\nLoading FictionalQA from HF Hub", flush=True)
    all_items = load_fictionalqa(allowed_styles=args.styles)

    if args.smoke_test:
        questions = all_items[:1]
        print("\n*** SMOKE TEST: 1 fiction ***", flush=True)
    else:
        questions = sample_items(all_items, args.n_fictions, args.seed)
        print(f"\nUsing {len(questions)} fictions (sampled)", flush=True)

    done = load_done_keys(args.output)
    if done:
        print(f"\nResume: {len(done)} chains already in {args.output} — will skip them.", flush=True)

    total_chains = len(questions) * sum(len(pool) for pool in ALL_INSTRUCTIONS.values())
    print(f"\nPlan: {len(questions)} fictions × 4 instructions × 3 wordings = {total_chains} chains")
    print(f"      each chain = {args.n_iterations} steps + 1 baseline = {args.n_iterations+1} rows")

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
                        "style": q["style"],
                        "event_id": q["event_id"],
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
