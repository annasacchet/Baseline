"""
Answer F1 evaluation for the 600q rewriting chains.

Same QA pipeline as scripts/15q/answer_f1_eval.py:
  QA model = allenai/OLMo-2-1124-32B-Instruct
  Prompt   = MuSiQue-style "answer based on context"
  Metric   = MuSiQue Answer F1 (SQuAD normalization, best over aliases)

Extras vs. the 15q script:
  --qid QID            restrict evaluation to a single qid (repeatable).
                       Useful while we have OFS recall + F1 for one qid only.
  --runs / --steps     further restrict by run / step set.
  --resume             skip (qid, group, instruction_type, run, step) rows
                       already present in the output CSV.

Default I/O paths point at results/600q/. Run on the GPU server — needs the
MuSiQue dev jsonl, which is gitignored locally.
"""

import argparse
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHAINS_CSV = REPO_ROOT / "results" / "600q" / "rewriting_chains_musique_600q.csv"
DEFAULT_MUSIQUE_PATH = REPO_ROOT / "musique_ans_v1.0_dev.jsonl"

QA_MODEL_ID = "allenai/OLMo-2-1124-32B-Instruct"
CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]


QA_USER_TEMPLATE = """Answer the question based on the context below. Give a short, direct answer — a few words at most, no explanation.

Context:
{context}

Question: {question}
Answer:"""


def load_musique_index(path: Path) -> dict:
    index = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            index[rec["id"]] = {
                "question": rec["question"],
                "answer": rec["answer"],
                "aliases": rec.get("answer_aliases") or [],
            }
    return index


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", text)

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
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def best_f1(pred, gold, aliases):
    candidates = [gold] + [a for a in aliases if a]
    best = 0.0
    best_ref = gold
    for ref in candidates:
        s = compute_f1(ref, pred)
        if s > best:
            best = s
            best_ref = ref
    return best, best_ref


def load_model(model_id, use_4bit):
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


def build_prompts(tokenizer, rows, musique):
    prompts = []
    for row in rows:
        ref = musique[row["qid"]]
        user = QA_USER_TEMPLATE.format(
            context=row["text"].strip(),
            question=ref["question"].strip(),
        )
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        ))
    return prompts


@torch.no_grad()
def generate_batch(tokenizer, model, prompts, max_new_tokens=64):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_tokens = out[:, enc["input_ids"].shape[1]:]
    return [t.strip() for t in tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)]


def main():
    ap = argparse.ArgumentParser(description="Answer F1 on 600q rewriting chains, with --qid filter.")
    ap.add_argument("--input", type=Path, default=DEFAULT_CHAINS_CSV)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_MUSIQUE_PATH)
    ap.add_argument("--model", default=QA_MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--qid", action="append", default=None,
                    help="Restrict to these qids (repeatable). Default: all qids in --input.")
    ap.add_argument("--runs", type=int, nargs="+", default=None)
    ap.add_argument("--steps", type=int, nargs="+", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Skip rows already present in the output CSV.")
    args = ap.parse_args()

    chains_csv = args.input
    qid_tag = ("_" + args.qid[0]) if (args.qid and len(args.qid) == 1) else ""
    output_csv = args.output or chains_csv.with_name(chains_csv.stem + f"_answer_f1{qid_tag}.csv")

    if not chains_csv.exists():
        raise FileNotFoundError(chains_csv)
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    print("Loading MuSiQue index ...")
    musique = load_musique_index(args.dataset)
    print(f"  {len(musique)} questions indexed")

    df = pd.read_csv(chains_csv).sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    if args.qid:
        df = df[df["qid"].isin(args.qid)]
    if args.runs is not None:
        df = df[df["run"].isin(args.runs)]
    if args.steps is not None:
        df = df[df["step"].isin(args.steps)]

    missing = set(df["qid"].unique()) - set(musique.keys())
    if missing:
        raise RuntimeError(f"qid in CSV but missing in MuSiQue: {missing}")

    if df.empty:
        raise RuntimeError("No rows match the filters.")

    # Dedupe E_0: same source text repeats across instruction_type for a (qid, run).
    e0_mask = df["step"] == 0
    e0_dedup = df[e0_mask].drop_duplicates(subset=["qid", "run"], keep="first")
    to_eval = pd.concat([e0_dedup, df[~e0_mask]], ignore_index=True)
    to_eval = to_eval.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    if args.resume and output_csv.exists():
        prev = pd.read_csv(output_csv)
        done_keys = {tuple(r[k] for k in CHAIN_KEYS + ["step"]) for _, r in prev.iterrows()}
        before = len(to_eval)
        # E_0 done check uses (qid, run, step=0), broadcasting later restores per-chain rows.
        def is_done(row):
            if row["step"] == 0:
                return any(
                    k[0] == row["qid"] and k[3] == row["run"] and k[4] == 0
                    for k in done_keys
                )
            return tuple(row[k] for k in CHAIN_KEYS + ["step"]) in done_keys
        to_eval = to_eval[~to_eval.apply(is_done, axis=1)].reset_index(drop=True)
        print(f"Resume: {before - len(to_eval)} rows already in {output_csv.name}, "
              f"{len(to_eval)} to go.")

    total = len(to_eval)
    if total == 0:
        print("Nothing left to evaluate. Done.")
        return

    print(f"Answer F1 on {total} rows — model={args.model}, batch={args.batch_size}")
    if args.qid:
        print(f"  qids: {args.qid}")

    tokenizer, model = load_model(args.model, args.use_4bit)

    rows = to_eval.to_dict(orient="records")
    results = []
    t_start = time.time()

    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        prompts = build_prompts(tokenizer, batch, musique)
        preds = generate_batch(tokenizer, model, prompts, max_new_tokens=64)

        for row, pred in zip(batch, preds):
            ref = musique[row["qid"]]
            f1, matched_ref = best_f1(pred, ref["answer"], ref["aliases"])
            results.append({
                **{k: row[k] for k in CHAIN_KEYS},
                "step": int(row["step"]),
                "question": ref["question"],
                "gold_answer": ref["answer"],
                "predicted_answer": pred,
                "matched_reference": matched_ref,
                "answer_f1": f1,
            })
            n_done = len(results)
            avg = (time.time() - t_start) / max(n_done, 1)
            eta = (total - n_done) * avg
            label = f"{row['qid']}/{row['group']}/{row['instruction_type']}/run{row['run']}/step{row['step']}"
            pred_short = (pred[:50] + "...") if len(pred) > 50 else pred
            print(f"[{n_done}/{total}] {label}  pred={pred_short!r:55s} "
                  f"gold={ref['answer']!r:25s} F1={f1:.3f}  ETA {eta/60:.1f} min")

    elapsed = time.time() - t_start
    print(f"\nTime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    results_df = pd.DataFrame(results)

    # Broadcast E_0 results to all (group, instruction_type) of the same (qid, run).
    if not results_df.empty:
        step0 = results_df[results_df["step"] == 0]
        step_gt0 = results_df[results_df["step"] > 0]
        if not step0.empty:
            all_chains = df[CHAIN_KEYS].drop_duplicates()
            step0_broadcast = all_chains.merge(
                step0.drop(columns=["group", "instruction_type"]),
                on=["qid", "run"], how="inner",
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
        columns="step", values="answer_f1", aggfunc="mean",
    )
    print(pivot.round(3))


if __name__ == "__main__":
    main()
