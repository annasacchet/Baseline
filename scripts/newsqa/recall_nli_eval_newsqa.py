"""
Recall evaluation via NLI for NewsQA rewriting chains.

For each (qid, instruction_type, run, step), takes the E_0 atomic facts
from the OFS details file and checks whether each fact is entailed by the
rewritten text E_t using a lightweight NLI cross-encoder (CPU-friendly).

Model: cross-encoder/nli-deberta-v3-small

Output:
  results/newsqa/rewriting_chains_newsqa_100q_recall_nli.csv
    — one row per (qid, instruction_type, run, step, fact)
    — columns: qid, group, instruction_type, run, step, fact, label (SURVIVED/LOST), score
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import CrossEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETAILS = REPO_ROOT / "results/newsqa/rewriting_chains_newsqa_100q_openfactscore_details.csv"
DEFAULT_CHAINS  = REPO_ROOT / "results/newsqa/rewriting_chains_newsqa_100q.csv"
DEFAULT_OUTPUT  = REPO_ROOT / "results/newsqa/rewriting_chains_newsqa_100q_recall_nli.csv"

NLI_MODEL  = "cross-encoder/nli-deberta-v3-small"
ENTAIL_IDX = 1   # index of 'entailment' label in model output
BATCH_SIZE = 32
CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--chains",  type=Path, default=DEFAULT_CHAINS)
    parser.add_argument("--output",  type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Entailment logit threshold (default: 0.0 = raw logit > 0)")
    parser.add_argument("--qid", action="append", default=None,
                        help="Restrict to one or more qid values (repeatable).")
    args = parser.parse_args()

    details = pd.read_csv(args.details, low_memory=False)
    chains  = pd.read_csv(args.chains)

    # E_0 facts: one set per qid (stored once, under elaborate/run0)
    e0 = details[details["step"] == 0][["qid", "fact"]].drop_duplicates()
    print(f"E_0 facts: {len(e0)} across {e0['qid'].nunique()} qid")

    # rewritten texts: step 1,2,3
    texts = chains[chains["step"] > 0][CHAIN_KEYS + ["step", "text"]].copy()

    if args.qid:
        e0    = e0[e0["qid"].isin(args.qid)]
        texts = texts[texts["qid"].isin(args.qid)]

    # resume
    done = set()
    if args.output.exists():
        prev = pd.read_csv(args.output)
        done = set(zip(prev["qid"], prev["instruction_type"], prev["run"],
                       prev["step"], prev["fact"]))
        print(f"Resume: {len(done)} rows already done.")

    print(f"Loading NLI model: {NLI_MODEL} ...")
    model = CrossEncoder(NLI_MODEL)
    print("Model loaded.")

    rows_out = []
    t_start  = time.time()
    total    = len(texts) * 0  # estimated below after groupby

    # iterate over each (qid, instruction_type, run, step)
    groups = list(texts.groupby(CHAIN_KEYS + ["step"]))
    total  = len(groups)

    for i, (key, grp) in enumerate(groups, 1):
        qid, group, instr, run, step = key
        text = grp["text"].iloc[0]

        facts_for_qid = e0[e0["qid"] == qid]["fact"].tolist()
        if not facts_for_qid:
            continue

        # filter already done facts
        facts_todo = [f for f in facts_for_qid
                      if (qid, instr, run, step, f) not in done]
        if not facts_todo:
            continue

        # NLI: premise = rewritten text, hypothesis = fact
        pairs = [(text, fact) for fact in facts_todo]
        logits = model.predict(pairs, batch_size=BATCH_SIZE)

        for fact, logit_row in zip(facts_todo, logits):
            entail_score = float(logit_row[ENTAIL_IDX])
            label = "SURVIVED" if entail_score > args.threshold else "LOST"
            rows_out.append({
                "qid": qid,
                "group": group,
                "instruction_type": instr,
                "run": run,
                "step": step,
                "fact": fact,
                "label": label,
                "entail_score": round(entail_score, 4),
            })

        # flush every 10 chains
        if rows_out and i % 10 == 0:
            _flush(rows_out, args.output)
            rows_out = []

        elapsed = time.time() - t_start
        eta = (elapsed / i) * (total - i)
        n_survived = sum(1 for r in rows_out if r["label"] == "SURVIVED")
        print(f"[{i}/{total}] {instr}/run{run}/step{step} qid={qid[-20:]}  "
              f"facts={len(facts_todo)}  ETA {eta/60:.1f}min", flush=True)

    if rows_out:
        _flush(rows_out, args.output)

    print(f"\nDone in {(time.time()-t_start)/60:.1f} min")
    print(f"Saved: {args.output}")

    # summary
    out = pd.read_csv(args.output)
    recall = out.groupby(["instruction_type", "step"]).apply(
        lambda x: (x["label"] == "SURVIVED").mean()
    ).unstack(level=0).round(3)
    print("\n=== Recall (% E_0 facts survived) ===")
    print(recall.to_string())


def _flush(rows, path):
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


if __name__ == "__main__":
    main()
