"""
Add BERTScore (Baseline and Consecutive modes) to FactScore evaluation.

§6.2 of the brainstorming defines two BERTScore modes:
  - Baseline:    sim(E_t, E_0)   — cumulative drift from the original
  - Consecutive: sim(E_t, E_{t-1}) — change between adjacent steps

Both are computed here. Baseline is what this script used to produce; Consecutive
is added so each step has a "how much did this step change the text" signal,
independent from the cumulative drift.

Output schema (additive — extends the previous file):
  bert_precision_baseline, bert_recall_baseline, bert_f1_baseline
  bert_precision_consecutive, bert_recall_consecutive, bert_f1_consecutive
"""

import time
from pathlib import Path

import pandas as pd
import torch
from bert_score import score as compute_bert_score

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "results" / "rewriting_chains_9q.csv"
FACTSCORE_PATH = REPO_ROOT / "results" / "rewriting_chains_9q_factscore.csv"
OUTPUT_PATH = REPO_ROOT / "results" / "rewriting_chains_9q_factscore_bertscore.csv"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

BERT_MODEL = "roberta-large"
BERT_NUM_LAYERS = 17
BERT_LANG = "en"
BERT_BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_step_index(chains_df: pd.DataFrame) -> dict:
    """Return {(qid, group, instr_type, run, step): text} for every row in the chain CSV.

    Used to look up both step 0 (Baseline reference) and step t-1 (Consecutive reference).
    """
    idx = {}
    for _, row in chains_df.iterrows():
        key = (
            row["qid"], row["group"], row["instruction_type"],
            int(row["run"]), int(row["step"]),
        )
        idx[key] = row["text"]
    return idx


def compute_bertscore_pairs(candidates, references):
    """Run BERTScore on aligned (candidate, reference) lists. Returns (P, R, F1) lists."""
    if not candidates:
        return [], [], []
    P, R, F1 = compute_bert_score(
        candidates,
        references,
        lang=BERT_LANG,
        model_type=BERT_MODEL,
        num_layers=BERT_NUM_LAYERS,
        batch_size=BERT_BATCH_SIZE,
        device=DEVICE,
        verbose=False,
    )
    return P.tolist(), R.tolist(), F1.tolist()


def main():
    print("=" * 70)
    print(f"BERTScore (Baseline + Consecutive)  —  device={DEVICE}")
    print("=" * 70)

    if not FACTSCORE_PATH.exists():
        raise FileNotFoundError(
            f"FactScore CSV not found: {FACTSCORE_PATH}\nRun factscore_eval.py first."
        )
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Chains CSV not found: {CSV_PATH}")

    print(f"\nLoading chains: {CSV_PATH}")
    chains_df = pd.read_csv(CSV_PATH)
    step_index = build_step_index(chains_df)
    print(f"  {len(step_index)} (chain, step) entries indexed")

    print(f"Loading FactScore: {FACTSCORE_PATH}")
    factscore_df = pd.read_csv(FACTSCORE_PATH)
    print(f"  {len(factscore_df)} rows to score")

    candidates: list = []
    refs_baseline: list = []
    refs_consecutive: list = []
    has_consecutive: list = []
    skipped_baseline = 0
    skipped_consecutive = 0

    for _, row in factscore_df.iterrows():
        qid, group, itype, run, step = (
            row["qid"], row["group"], row["instruction_type"],
            int(row["run"]), int(row["step"]),
        )
        cand = step_index.get((qid, group, itype, run, step))
        ref_b = step_index.get((qid, group, itype, run, 0))
        ref_c = step_index.get((qid, group, itype, run, step - 1))

        if cand is None or ref_b is None:
            candidates.append("")
            refs_baseline.append("")
            refs_consecutive.append("")
            has_consecutive.append(False)
            skipped_baseline += 1
            continue

        candidates.append(cand)
        refs_baseline.append(ref_b)

        if ref_c is not None and step >= 1:
            refs_consecutive.append(ref_c)
            has_consecutive.append(True)
        else:
            refs_consecutive.append("")
            has_consecutive.append(False)
            skipped_consecutive += 1

    valid_baseline_idx = [i for i, c in enumerate(candidates) if c and refs_baseline[i]]
    valid_consec_idx = [i for i, _ in enumerate(candidates) if has_consecutive[i] and candidates[i]]

    print(f"\nBaseline pairs:    {len(valid_baseline_idx)} (skipped: {skipped_baseline})")
    print(f"Consecutive pairs: {len(valid_consec_idx)} (skipped: {skipped_consecutive})")
    print(f"BERTScore model: {BERT_MODEL} (layer {BERT_NUM_LAYERS}), batch={BERT_BATCH_SIZE}")

    # Baseline mode: sim(E_t, E_0)
    print("\nComputing Baseline BERTScore...")
    t0 = time.time()
    cand_b = [candidates[i] for i in valid_baseline_idx]
    ref_b = [refs_baseline[i] for i in valid_baseline_idx]
    P_b, R_b, F1_b = compute_bertscore_pairs(cand_b, ref_b)
    print(f"  done in {time.time() - t0:.1f}s")

    # Consecutive mode: sim(E_t, E_{t-1})
    print("Computing Consecutive BERTScore...")
    t0 = time.time()
    cand_c = [candidates[i] for i in valid_consec_idx]
    ref_c = [refs_consecutive[i] for i in valid_consec_idx]
    P_c, R_c, F1_c = compute_bertscore_pairs(cand_c, ref_c)
    print(f"  done in {time.time() - t0:.1f}s")

    bp_b = [None] * len(factscore_df)
    br_b = [None] * len(factscore_df)
    bf_b = [None] * len(factscore_df)
    bp_c = [None] * len(factscore_df)
    br_c = [None] * len(factscore_df)
    bf_c = [None] * len(factscore_df)

    for k, i in enumerate(valid_baseline_idx):
        bp_b[i], br_b[i], bf_b[i] = P_b[k], R_b[k], F1_b[k]
    for k, i in enumerate(valid_consec_idx):
        bp_c[i], br_c[i], bf_c[i] = P_c[k], R_c[k], F1_c[k]

    out = factscore_df.copy()
    out["bert_precision_baseline"] = bp_b
    out["bert_recall_baseline"] = br_b
    out["bert_f1_baseline"] = bf_b
    out["bert_precision_consecutive"] = bp_c
    out["bert_recall_consecutive"] = br_c
    out["bert_f1_consecutive"] = bf_c

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")

    print("\n" + "=" * 70)
    print("BERTScore F1 — mean per (group, instruction_type, step)")
    print("=" * 70)
    pivot_b = out.pivot_table(
        index=["group", "instruction_type"],
        columns="step",
        values="bert_f1_baseline",
        aggfunc="mean",
    )
    print("\nBaseline (vs E_0):")
    print(pivot_b.round(3))

    pivot_c = out.pivot_table(
        index=["group", "instruction_type"],
        columns="step",
        values="bert_f1_consecutive",
        aggfunc="mean",
    )
    print("\nConsecutive (vs E_{t-1}):")
    print(pivot_c.round(3))

    print("\n" + "=" * 70)
    print("Pearson correlations")
    print("=" * 70)
    valid = out["bert_f1_baseline"].notna() & out["init_score"].notna()
    if valid.sum() > 0:
        r = out[valid][["init_score", "bert_f1_baseline"]].corr().iloc[0, 1]
        print(f"  FactScore vs BERTScore (Baseline):    r = {r:.4f}")
    valid = out["bert_f1_consecutive"].notna() & out["init_score"].notna()
    if valid.sum() > 0:
        r = out[valid][["init_score", "bert_f1_consecutive"]].corr().iloc[0, 1]
        print(f"  FactScore vs BERTScore (Consecutive): r = {r:.4f}")


if __name__ == "__main__":
    main()
