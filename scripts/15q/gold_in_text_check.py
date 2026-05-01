"""
Cross-check: is the gold answer literally present in the rewritten text?

For each row in the Answer F1 output CSV, joins with the rewriting chains CSV
to check whether the normalized gold answer appears as a substring of the
normalized E_t text. This separates:

  gold_in_text=True  & F1=0  → false negative (morphology/normalization mismatch)
  gold_in_text=False & F1=0  → real degradation (rewriting removed the fact)
  gold_in_text=False & F1>0  → model answered from parametric memory, not context
  gold_in_text=True  & F1>0  → clean hit

Usage
-----
    python3 scripts/15q/gold_in_text_check.py
    python3 scripts/15q/gold_in_text_check.py \
        --f1-csv results/15q/rewriting_chains_15q_answer_f1.csv \
        --chains-csv results/15q/rewriting_chains_15q.csv \
        --output results/15q/gold_in_text_check.csv
"""

import argparse
import re
import string
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_F1_CSV = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q_answer_f1.csv"
DEFAULT_CHAINS_CSV = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q.csv"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run", "step"]


# ---------------------------------------------------------------------------
# Same normalization as answer_f1_eval.py / MuSiQue official
# ---------------------------------------------------------------------------

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def gold_in_text(gold: str, text: str) -> bool:
    """True if any gold token sequence appears verbatim in the normalized text."""
    return normalize_answer(gold) in normalize_answer(text)


def gold_in_text_any(gold: str, aliases: list, text: str) -> bool:
    """True if gold OR any alias appears in the normalized text."""
    candidates = [gold] + [a for a in aliases if a]
    return any(normalize_answer(c) in normalize_answer(text) for c in candidates)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-check gold answer presence in rewritten texts.")
    parser.add_argument("--f1-csv", type=Path, default=DEFAULT_F1_CSV,
                        help=f"Answer F1 output CSV (default: {DEFAULT_F1_CSV}).")
    parser.add_argument("--chains-csv", type=Path, default=DEFAULT_CHAINS_CSV,
                        help=f"Rewriting chains CSV with text column (default: {DEFAULT_CHAINS_CSV}).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path. Default: derived from --f1-csv (stem + '_gold_in_text.csv').")
    args = parser.parse_args()

    f1_csv = args.f1_csv
    chains_csv = args.chains_csv
    output_csv = args.output or f1_csv.with_name(f1_csv.stem + "_gold_in_text.csv")

    if not f1_csv.exists():
        raise FileNotFoundError(f"Answer F1 CSV not found: {f1_csv}")
    if not chains_csv.exists():
        raise FileNotFoundError(f"Chains CSV not found: {chains_csv}")

    f1_df = pd.read_csv(f1_csv)
    chains_df = pd.read_csv(chains_csv)

    # Keep only the text column from chains for the join
    text_df = chains_df[CHAIN_KEYS + ["text"]].drop_duplicates(subset=CHAIN_KEYS)

    merged = f1_df.merge(text_df, on=CHAIN_KEYS, how="left")

    missing_text = merged["text"].isna().sum()
    if missing_text > 0:
        print(f"WARNING: {missing_text} rows could not be joined with a text — check CHAIN_KEYS alignment.")

    merged["gold_norm"] = merged["gold_answer"].apply(normalize_answer)
    merged["text_norm"] = merged["text"].fillna("").apply(normalize_answer)
    merged["gold_in_text"] = merged.apply(
        lambda r: r["gold_norm"] in r["text_norm"], axis=1
    )

    # Classify each row
    def classify(row):
        if row["gold_in_text"] and row["answer_f1"] > 0:
            return "hit"
        elif row["gold_in_text"] and row["answer_f1"] == 0:
            return "false_negative"
        elif not row["gold_in_text"] and row["answer_f1"] > 0:
            return "parametric_memory"
        else:
            return "degraded"

    merged["category"] = merged.apply(classify, axis=1)

    output_cols = CHAIN_KEYS + [
        "question", "gold_answer", "predicted_answer", "answer_f1",
        "gold_in_text", "category", "gold_norm", "text_norm",
    ]
    merged[output_cols].to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    print()

    # Summary table
    print("=" * 60)
    print("CATEGORY COUNTS")
    print("=" * 60)
    counts = merged["category"].value_counts()
    print(counts.to_string())
    print()

    print("=" * 60)
    print("CATEGORY BY STEP")
    print("=" * 60)
    pivot = merged.pivot_table(
        index="step", columns="category", values="answer_f1", aggfunc="count", fill_value=0
    )
    print(pivot.to_string())
    print()

    print("=" * 60)
    print("MEAN F1 BY STEP — full vs gold_in_text only")
    print("=" * 60)
    by_step = merged.groupby("step")["answer_f1"].mean().rename("mean_f1_all")
    by_step_present = (
        merged[merged["gold_in_text"]].groupby("step")["answer_f1"].mean().rename("mean_f1_gold_present")
    )
    print(pd.concat([by_step, by_step_present], axis=1).round(3).to_string())
    print()

    print("=" * 60)
    print("FALSE NEGATIVES (gold in text but F1=0)")
    print("=" * 60)
    fn = merged[merged["category"] == "false_negative"][
        ["qid", "group", "instruction_type", "run", "step", "gold_answer", "predicted_answer"]
    ]
    if fn.empty:
        print("  None found.")
    else:
        print(fn.to_string(index=False))


if __name__ == "__main__":
    main()
