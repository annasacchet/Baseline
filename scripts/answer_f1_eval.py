"""
Answer F1 evaluation for the rewriting chains.

QA model: OLMo-3.1-32B-Instruct, loaded in-process with transformers.
Designed to run on a GPU server: load the model once, iterate over all chain
steps, compute Answer F1 against the MuSiQue gold answer.

Pipeline
--------
For each (qid, group, instruction_type, run, step) in rewriting_chains32b.csv:
  1. load the text E_t (E_0 is the original evidence, E_1..E_3 are the rewrites)
  2. load the original MuSiQue question + gold answer from musique_ans_v1.0_dev.jsonl
     (indexed by qid — the question stays pinned to the chain)
  3. prompt OLMo with evidence + question, no system prompt, no extra instructions
  4. compute Answer F1 vs gold (+ aliases), MuSiQue-official normalization.

Why step 0 is kept: the brainstorming frames Answer F1 as a DROP over steps. The
drop is only meaningful relative to the baseline F1 measured on E_0 with the
same QA model — otherwise we only know the absolute F1 at t>0, not how much was
lost. Evaluating E_0 too gives us the per-chain ceiling.

Scope notes
-----------
- The MuSiQue dev file is only used to look up `question` and `answer` by qid.
  Paragraphs are NOT pulled from it — the context given to the QA model is the
  E_t from the rewriting chain, which is the point of the experiment.
- Any qid present in the rewriting CSV is resolved against MuSiQue by id, so
  the script already scales to the full dataset: flip TEST_MODE to False.
"""

import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHAINS_CSV = Path("/Users/annasacchet/Desktop/Baseline/results/rewriting_chains32b.csv")
MUSIQUE_PATH = Path("/Users/annasacchet/Desktop/musique_ans_v1.0_dev.jsonl")
OUTPUT_CSV = Path("/Users/annasacchet/Desktop/Baseline/results/rewriting_chains32b_answer_f1.csv")

QA_MODEL_ID = "allenai/Olmo-3.1-32B-Instruct"
CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

# Generation config: deterministic, short answers.
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0
BATCH_SIZE = 4

# Pilot mode: keep the baseline scope (1 question, run 0) to match the current
# FactScore/BERTScore runs. Flip TEST_MODE to False to evaluate every chain.
TEST_MODE = True
TEST_FILTER = {"qid": "2hop__635544_110949", "run": 0}


QA_USER_TEMPLATE = """{context}

{question}"""


# ---------------------------------------------------------------------------
# MuSiQue lookup
# ---------------------------------------------------------------------------

def load_musique_index(path: Path) -> dict:
    """Return {qid: {'question': str, 'answer': str, 'aliases': [str]}}."""
    index = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["id"]
            index[qid] = {
                "question": rec["question"],
                "answer": rec["answer"],
                "aliases": rec.get("answer_aliases") or [],
            }
    return index


# ---------------------------------------------------------------------------
# Answer F1 — verbatim from MuSiQue's metrics/answer.py
# (StonyBrookNLP/musique, SQuAD-style normalization)
# ---------------------------------------------------------------------------

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def best_f1(pred, gold, aliases):
    """max F1 over {gold} ∪ aliases — matches evaluate_v1.0.py."""
    candidates = [gold] + [a for a in aliases if a]
    best = 0.0
    best_ref = gold
    for ref in candidates:
        s = compute_f1(ref, pred)
        if s > best:
            best = s
            best_ref = ref
    return best, best_ref


# ---------------------------------------------------------------------------
# OLMo model wrapper
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # needed for decoder-only batched generation

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  device map: {getattr(model, 'hf_device_map', 'n/a')}")
    return tokenizer, model


def build_prompts(tokenizer, rows, musique):
    prompts = []
    for row in rows:
        ref = musique[row["qid"]]
        user = QA_USER_TEMPLATE.format(
            context=row["text"].strip(),
            question=ref["question"].strip(),
        )
        messages = [
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


@torch.no_grad()
def generate_batch(tokenizer, model, prompts):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=TEMPERATURE > 0.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    if TEMPERATURE > 0.0:
        gen_kwargs["temperature"] = TEMPERATURE

    out = model.generate(**enc, **gen_kwargs)
    # strip the prompt tokens from each sequence
    gen_tokens = out[:, enc["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return [t.strip() for t in texts]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not CHAINS_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {CHAINS_CSV}")
    if not MUSIQUE_PATH.exists():
        raise FileNotFoundError(f"File non trovato: {MUSIQUE_PATH}")

    print("Loading MuSiQue index...")
    musique = load_musique_index(MUSIQUE_PATH)
    print(f"  {len(musique)} questions indexed")

    df = pd.read_csv(CHAINS_CSV)
    df = df.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    qids_in_csv = set(df["qid"].unique())
    missing = qids_in_csv - set(musique.keys())
    if missing:
        raise RuntimeError(f"qid nel CSV non trovati in MuSiQue: {missing}")

    to_eval = df.copy()

    if TEST_MODE and TEST_FILTER:
        for k, v in TEST_FILTER.items():
            to_eval = to_eval[to_eval[k] == v]
        print(f"*** TEST MODE: filter={TEST_FILTER} -> {len(to_eval)} rows ***")

    # Dedupe E_0: same text is repeated across all instructions of a (qid, run).
    # We still want F1 at step 0 as the per-chain ceiling, but computing it once
    # per (qid, run) is enough. Keep the first occurrence; the output row is
    # later broadcast to all instruction_types of that (qid, run) so pivots align.
    e0_mask = to_eval["step"] == 0
    e0_dedup = to_eval[e0_mask].drop_duplicates(subset=["qid", "run"], keep="first")
    to_eval = pd.concat([e0_dedup, to_eval[~e0_mask]], ignore_index=True)
    to_eval = to_eval.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    if to_eval.empty:
        raise RuntimeError("Nessuna riga da valutare dopo il filtro.")

    total = len(to_eval)
    print(f"Answer F1 su {total} testi (incluso step 0) — QA model = {QA_MODEL_ID}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    tokenizer, model = load_model(QA_MODEL_ID)

    rows = to_eval.to_dict(orient="records")
    results = []
    t_start = time.time()

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        prompts = build_prompts(tokenizer, batch, musique)
        preds = generate_batch(tokenizer, model, prompts)

        for row, pred in zip(batch, preds):
            ref = musique[row["qid"]]
            f1, matched_ref = best_f1(pred, ref["answer"], ref["aliases"])
            out = {
                **{k: row[k] for k in CHAIN_KEYS},
                "step": int(row["step"]),
                "question": ref["question"],
                "gold_answer": ref["answer"],
                "predicted_answer": pred,
                "matched_reference": matched_ref,
                "answer_f1": f1,
            }
            results.append(out)
            label = f"{out['group']}/{out['instruction_type']}/run{out['run']}/step{out['step']}"
            pred_short = (pred[:50] + "...") if len(pred) > 50 else pred
            print(f"[{len(results)}/{total}] {label}  pred={pred_short!r:55s} gold={ref['answer']!r:25s} F1={f1:.3f}")

    elapsed = time.time() - t_start
    print(f"\nTempo totale: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    results_df = pd.DataFrame(results)

    # Broadcast E_0 predictions to every (group, instruction_type) of the same
    # (qid, run) — we only evaluated E_0 once per (qid, run), but for pivots
    # and per-chain plots we want a step=0 row for each chain.
    if not results_df.empty:
        step0 = results_df[results_df["step"] == 0]
        step_gt0 = results_df[results_df["step"] > 0]
        if not step0.empty:
            all_chains = df[CHAIN_KEYS].drop_duplicates()
            if TEST_MODE and TEST_FILTER:
                for k, v in TEST_FILTER.items():
                    all_chains = all_chains[all_chains[k] == v]
            step0_broadcast = all_chains.merge(
                step0.drop(columns=["group", "instruction_type"]),
                on=["qid", "run"],
                how="inner",
            )
            results_df = pd.concat([step0_broadcast, step_gt0], ignore_index=True)
            results_df = results_df.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    if OUTPUT_CSV.exists():
        prev = pd.read_csv(OUTPUT_CSV)
        merged = pd.concat([prev, results_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=CHAIN_KEYS + ["step"], keep="last")
        merged.to_csv(OUTPUT_CSV, index=False)
    else:
        results_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved: {OUTPUT_CSV}")

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
