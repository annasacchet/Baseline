"""
Answer F1 evaluation for NewsQA rewriting chains.

QA model: OLMo-3.1-32B-Instruct (loaded in-process with transformers).
Designed to run on a GPU server: load the model once, iterate over all chain
steps, compute Answer F1 against the NewsQA validated answers.

Differences from the MuSiQue version
-------------------------------------
- The chains CSV already contains `question`, `gold_answer`, and
  `gold_answer_aliases` (||-joined string of all validated extractive spans
  with crowdsourcer agreement >= 2). No external dataset file is needed.
- F1 is computed as max over (gold + aliases), matching the official NewsQA /
  SQuAD evaluation, which compares each prediction to *every* human answer.
- The QA prompt instructs OLMo to extract a verbatim span from the context,
  matching the extractive nature of the original NewsQA evaluation.
- Generation `max_new_tokens` is bumped to 96 because NewsQA answers are
  longer on average than MuSiQue (clause/verb phrases are common — see paper
  Table 1: clause phrases account for 18.3% of answers).

Pipeline
--------
For each (qid, group, instruction_type, run, step) in the chains CSV:
  1. load the text E_t
  2. read question + gold_answer + aliases from the CSV row
  3. prompt OLMo with evidence + question
  4. compute max F1 over all gold answer strings

E_0 is deduplicated across instruction_types of the same (qid, run) and then
broadcast back, matching the MuSiQue version's behaviour.
"""

from __future__ import annotations

import argparse
import re
import string
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHAINS_CSV = REPO_ROOT / "results" / "newsqa" / "rewriting_chains_newsqa.csv"

QA_MODEL_ID = "allenai/OLMo-3.1-32B-Instruct"
CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]
ALIAS_SEP = "||"

QA_USER_TEMPLATE = """Answer the question using only words copied verbatim from the context below. \
Your answer must be a continuous span of text that appears exactly in the context — do not paraphrase, \
do not add words not in the context.

Context:
{context}

Question: {question}
Answer (verbatim span from context):"""


# ---------------------------------------------------------------------------
# Answer F1 — SQuAD-style normalization (same as NewsQA official)
# ---------------------------------------------------------------------------

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def parse_aliases(value, fallback_gold: str) -> list[str]:
    """Pull the alias list out of the chain CSV. Always include the gold answer."""
    aliases: list[str] = []
    if isinstance(value, str) and value.strip():
        aliases = [a.strip() for a in value.split(ALIAS_SEP) if a and a.strip()]
    if fallback_gold and fallback_gold not in aliases:
        aliases.insert(0, fallback_gold)
    return aliases


def best_f1(pred: str, golds: list[str]) -> tuple[float, str]:
    """Return (max F1, the gold string that produced it)."""
    best_score = 0.0
    best_ref = golds[0] if golds else ""
    for g in golds:
        s = compute_f1(g, pred)
        if s > best_score:
            best_score = s
            best_ref = g
    return best_score, best_ref


# ---------------------------------------------------------------------------
# OLMo model wrapper
# ---------------------------------------------------------------------------

def load_model(model_id: str, use_4bit: bool):
    print(f"Loading {model_id} (4-bit={use_4bit}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs = {"device_map": "auto"}
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
    print(f"  device map: {getattr(model, 'hf_device_map', 'n/a')}")
    return tokenizer, model


def build_prompts(tokenizer, rows):
    prompts = []
    for row in rows:
        user = QA_USER_TEMPLATE.format(
            context=str(row["text"]).strip(),
            question=str(row["question"]).strip(),
        )
        messages = [{"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


@torch.no_grad()
def generate_batch(tokenizer, model, prompts, max_new_tokens: int = 96):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_tokens = out[:, enc["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return [t.strip() for t in texts]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Answer F1 evaluation on NewsQA rewriting chains.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CHAINS_CSV,
                        help=f"Input chains CSV (default: {DEFAULT_CHAINS_CSV}).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path. Default: <input_stem>_answer_f1.csv.")
    parser.add_argument("--model", default=QA_MODEL_ID,
                        help=f"HF model id for QA (default: {QA_MODEL_ID}).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Generation batch size (default: 8).")
    # NewsQA answers include long clause/verb phrases (paper Table 1 — 18.3%
    # clause phrases). Bumped from 64 to 96 to avoid truncating.
    parser.add_argument("--max-new-tokens", type=int, default=96,
                        help="QA generation max new tokens (default: 96).")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Enable 4-bit NF4 quantization. Default: bfloat16.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run only on the first qid, run 0.")
    args = parser.parse_args()

    chains_csv = args.input
    output_csv = args.output or chains_csv.with_name(chains_csv.stem + "_answer_f1_span.csv")

    if not chains_csv.exists():
        raise FileNotFoundError(f"File not found: {chains_csv}")

    df = pd.read_csv(chains_csv)
    if "gold_answer_aliases" not in df.columns:
        df["gold_answer_aliases"] = ""
    df = df.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    to_eval = df.copy()

    if args.smoke_test:
        first_qid = df["qid"].iloc[0]
        to_eval = to_eval[(to_eval["qid"] == first_qid) & (to_eval["run"] == 0)]
        print(f"*** SMOKE TEST: {len(to_eval)} rows for qid={first_qid} ***")

    # Deduplicate E_0 across instruction_types of the same (qid, run)
    e0_mask = to_eval["step"] == 0
    e0_dedup = to_eval[e0_mask].drop_duplicates(subset=["qid", "run"], keep="first")
    to_eval = pd.concat([e0_dedup, to_eval[~e0_mask]], ignore_index=True)
    to_eval = to_eval.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    if to_eval.empty:
        raise RuntimeError("No rows to evaluate.")

    total = len(to_eval)
    print(f"Answer F1 on {total} texts — QA model = {args.model}")
    print(f"Batch size: {args.batch_size}")

    tokenizer, model = load_model(args.model, args.use_4bit)

    rows = to_eval.to_dict(orient="records")
    results = []
    t_start = time.time()

    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        prompts = build_prompts(tokenizer, batch)
        preds = generate_batch(tokenizer, model, prompts, max_new_tokens=args.max_new_tokens)

        for row, pred in zip(batch, preds):
            gold = str(row["gold_answer"])
            aliases = parse_aliases(row.get("gold_answer_aliases"), gold)
            f1, matched_ref = best_f1(pred, aliases)
            out = {
                **{k: row[k] for k in CHAIN_KEYS},
                "step": int(row["step"]),
                "question": row["question"],
                "gold_answer": gold,
                "gold_answer_aliases": ALIAS_SEP.join(aliases),
                "predicted_answer": pred,
                "matched_reference": matched_ref,
                "answer_f1": f1,
            }
            results.append(out)
            label = f"{out['group']}/{out['instruction_type']}/run{out['run']}/step{out['step']}"
            pred_short = (pred[:50] + "...") if len(pred) > 50 else pred
            n_done = len(results)
            avg = (time.time() - t_start) / max(n_done, 1)
            eta = (total - n_done) * avg
            print(
                f"[{n_done}/{total}] {label}  pred={pred_short!r:55s} gold={gold!r:25s} "
                f"F1={f1:.3f}  ETA {eta/60:.1f} min"
            )

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    results_df = pd.DataFrame(results)

    # Broadcast E_0 predictions to every (group, instruction_type) of the same (qid, run)
    if not results_df.empty:
        step0 = results_df[results_df["step"] == 0]
        step_gt0 = results_df[results_df["step"] > 0]
        if not step0.empty:
            all_chains = df[CHAIN_KEYS].drop_duplicates()
            if args.smoke_test:
                first_qid = df["qid"].iloc[0]
                all_chains = all_chains[(all_chains["qid"] == first_qid) & (all_chains["run"] == 0)]
            step0_broadcast = all_chains.merge(
                step0.drop(columns=["group", "instruction_type"]),
                on=["qid", "run"],
                how="inner",
            )
            results_df = pd.concat([step0_broadcast, step_gt0], ignore_index=True)
            results_df = results_df.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        prev = pd.read_csv(output_csv)
        merged = pd.concat([prev, results_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=CHAIN_KEYS + ["step"], keep="last")
        merged.to_csv(output_csv, index=False)
    else:
        results_df.to_csv(output_csv, index=False)

    print(f"\nSaved: {output_csv}")

    print()
    print("=" * 70)
    print("ANSWER F1 — mean per (group, instruction_type, step)")
    print("=" * 70)
    pivot = results_df.pivot_table(
        index=["group", "instruction_type"],
        columns="step",
        values="answer_f1",
        aggfunc="mean",
    )
    print(pivot.round(3))


if __name__ == "__main__":
    main()
