"""Answer F1 on MuSiQue rewriting chains — QA model = OLMo-3.1-32B via vLLM.

Same prompt / aliases lookup / SQuAD-style F1 as the 600q pipeline.
Generation is done with vLLM (TP=2, bf16) for speed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gptoss120b"))
from _common.f1_utils import best_f1  # noqa: E402
from _common.olmo_constants import CHAIN_KEYS, OLMO_MODEL_ID  # noqa: E402
def _load_backend(name):
    """Return backend funcs from the vLLM or HF module (same names in both)."""
    if name == "hf":
        from _common.olmo_hf import (
            generate_batch_vllm, hf_login_if_token, load_vllm, render_chat,
        )
    else:
        from _common.olmo_vllm import (
            generate_batch_vllm, hf_login_if_token, load_vllm, render_chat,
        )
    return load_vllm, render_chat, generate_batch_vllm, hf_login_if_token

DEFAULT_CHAINS_CSV = REPO_ROOT / "results" / "gptoss120b" / "musique" / "rewriting_chains_musique.csv"
DEFAULT_MUSIQUE_PATH = REPO_ROOT / "musique_ans_v1.0_dev.jsonl"

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


def build_prompts(tokenizer, rows, musique):
    prompts = []
    for row in rows:
        ref = musique[row["qid"]]
        user = QA_USER_TEMPLATE.format(
            context=str(row["text"]).strip(),
            question=ref["question"].strip(),
        )
        prompts.append(render_chat(tokenizer, None, user))
    return prompts


def main():
    ap = argparse.ArgumentParser(description="Answer F1 on MuSiQue chains, OLMo-3.1-32B QA via vLLM.")
    ap.add_argument("--input", type=Path, default=DEFAULT_CHAINS_CSV)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_MUSIQUE_PATH)
    ap.add_argument("--model", default=OLMO_MODEL_ID)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    ap.add_argument("--quantization", default=None,
                    help="vLLM quantization. Pass 'bitsandbytes' for on-the-fly NF4 4-bit.")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="Disable CUDA graph capture / torch.compile (avoids JIT C++ build failures).")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                    help="Inference backend. 'hf' = transformers 4-bit (robust on old CUDA drivers).")
    ap.add_argument("--qid", action="append", default=None)
    ap.add_argument("--runs", type=int, nargs="+", default=None)
    ap.add_argument("--steps", type=int, nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # Bind backend functions as module globals (build_prompts uses render_chat).
    global load_vllm, render_chat, generate_batch_vllm, hf_login_if_token
    load_vllm, render_chat, generate_batch_vllm, hf_login_if_token = _load_backend(args.backend)
    print(f"Backend: {args.backend}", flush=True)

    chains_csv = args.input
    output_csv = args.output or chains_csv.with_name(chains_csv.stem + "_answer_f1.csv")

    if not chains_csv.exists():
        raise FileNotFoundError(chains_csv)
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    hf_login_if_token()
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

    e0 = df[df["step"] == 0].drop_duplicates(subset=["qid", "run"], keep="first")
    to_eval = pd.concat([e0, df[df["step"] > 0]], ignore_index=True)
    to_eval = to_eval.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    if args.resume and output_csv.exists():
        prev = pd.read_csv(output_csv)
        done_keys = {tuple(r[k] for k in CHAIN_KEYS + ["step"]) for _, r in prev.iterrows()}
        before = len(to_eval)
        def is_done(row):
            if row["step"] == 0:
                return any(k[0] == row["qid"] and k[3] == row["run"] and k[4] == 0 for k in done_keys)
            return tuple(row[k] for k in CHAIN_KEYS + ["step"]) in done_keys
        to_eval = to_eval[~to_eval.apply(is_done, axis=1)].reset_index(drop=True)
        print(f"Resume: {before - len(to_eval)} rows already in {output_csv.name}, {len(to_eval)} to go.")

    total = len(to_eval)
    if total == 0:
        print("Nothing left to evaluate.")
        return

    print(f"Answer F1 on {total} rows — model={args.model}")
    llm, tokenizer = load_vllm(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        quantization=args.quantization,
        enforce_eager=args.enforce_eager,
    )

    # vLLM is happy with one giant batch — it does its own continuous batching.
    rows = to_eval.to_dict(orient="records")
    print(f"Building {total} prompts ...")
    prompts = build_prompts(tokenizer, rows, musique)

    print("Generating ...")
    t0 = time.time()
    preds = generate_batch_vllm(llm, prompts, temperature=0.0, max_new_tokens=args.max_new_tokens)
    print(f"  vLLM generated {total} answers in {time.time() - t0:.1f}s")

    results = []
    for row, pred in zip(rows, preds):
        ref = musique[row["qid"]]
        golds = [ref["answer"]] + [a for a in ref["aliases"] if a]
        f1, matched_ref = best_f1(pred, golds)
        results.append({
            **{k: row[k] for k in CHAIN_KEYS},
            "step": int(row["step"]),
            "question": ref["question"], "gold_answer": ref["answer"],
            "predicted_answer": pred, "matched_reference": matched_ref,
            "answer_f1": f1,
        })

    results_df = pd.DataFrame(results)

    # Broadcast E_0 to all (group, instruction_type) of the same (qid, run)
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
    if output_csv.exists() and args.resume:
        prev = pd.read_csv(output_csv)
        merged = pd.concat([prev, results_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=CHAIN_KEYS + ["step"], keep="last")
        merged.to_csv(output_csv, index=False)
    else:
        results_df.to_csv(output_csv, index=False)

    print(f"\nSaved: {output_csv}")
    pivot = results_df.pivot_table(
        index=["group", "instruction_type"], columns="step", values="answer_f1", aggfunc="mean",
    )
    print(pivot.round(3))


if __name__ == "__main__":
    main()
