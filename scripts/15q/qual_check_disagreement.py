"""
Qualitative check on OFS vs FS disagreement.

For the chains where OpenFActScore and our GPT-4o-mini FActScore disagree the
most (in absolute init_score difference), print the NOT_SUPPORTED facts that
Gemma flagged, side-by-side with the source E_0. Lets the reviewer judge
whether each flag is a true hallucination or a judge false-positive.

Usage:
  python scripts/qual_check_disagreement.py
  python scripts/qual_check_disagreement.py --top 6 --max-facts 12
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

OFS_PATH = RESULTS / "15q" / "rewriting_chains_15q_openfactscore.csv"
OFS_DETAILS_PATH = RESULTS / "15q" / "rewriting_chains_15q_openfactscore_details.csv"
FS_PATH = RESULTS / "archive" / "rewriting_chains32b_factscore.csv"
SOURCE_PATH = RESULTS / "15q" / "rewriting_chains_15q.csv"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run", "step"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=4, help="How many top-disagreement chains to inspect.")
    parser.add_argument("--max-facts", type=int, default=10, help="Max NOT_SUPPORTED facts to print per chain.")
    args = parser.parse_args()

    ofs = pd.read_csv(OFS_PATH)
    fs = pd.read_csv(FS_PATH)
    details = pd.read_csv(OFS_DETAILS_PATH)
    src_df = pd.read_csv(SOURCE_PATH)

    # FS file only has run=0; align OFS to it
    ofs_r0 = ofs[ofs["run"] == 0]
    merged = ofs_r0[CHAIN_KEYS + ["init_score"]].merge(
        fs[CHAIN_KEYS + ["init_score"]],
        on=CHAIN_KEYS,
        suffixes=("_ofs", "_fs"),
    )
    merged["abs_delta"] = (merged["init_score_ofs"] - merged["init_score_fs"]).abs()
    merged = merged.sort_values("abs_delta", ascending=False).head(args.top)

    sources = (
        src_df[src_df["step"] == 0]
        .set_index(["qid", "group", "instruction_type", "run"])["text"]
        .to_dict()
    )
    rewrites = (
        src_df.set_index(CHAIN_KEYS)["text"].to_dict()
    )

    for _, row in merged.iterrows():
        chain = tuple(row[k] for k in CHAIN_KEYS)
        chain_key_no_step = chain[:4]

        print("=" * 80)
        print(f"{row['group']}/{row['instruction_type']}/run{row['run']}/step{row['step']}")
        print(f"  OFS init_score = {row['init_score_ofs']:.3f}")
        print(f"  FS  init_score = {row['init_score_fs']:.3f}")
        print(f"  |Δ|            = {row['abs_delta']:.3f}")
        print()

        source = sources.get(chain_key_no_step, "")
        print("--- SOURCE (E_0) ---")
        print(source[:1200] + ("..." if len(source) > 1200 else ""))
        print()

        rewrite = rewrites.get(chain, "")
        print("--- REWRITE (this step) ---")
        print(rewrite[:800] + ("..." if len(rewrite) > 800 else ""))
        print()

        sub = details[
            (details["qid"] == row["qid"])
            & (details["group"] == row["group"])
            & (details["instruction_type"] == row["instruction_type"])
            & (details["run"] == row["run"])
            & (details["step"] == row["step"])
        ]
        unsup = sub[sub["label"] == "NOT_SUPPORTED"]

        print(f"--- NOT_SUPPORTED facts flagged by Gemma ({len(unsup)} total, showing up to {args.max_facts}) ---")
        for i, (_, fr) in enumerate(unsup.head(args.max_facts).iterrows(), 1):
            print(f"  [{i}] {fr['fact']}")
        print()


if __name__ == "__main__":
    main()
