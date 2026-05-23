"""
F1 OpenFActScore for a single qid (default: 2hop__14092_8311).

Joins the forward OFS precision CSV with the recall CSV on
(qid, group, instruction_type, run, step) and computes:

  precision = factscore                (from forward OFS)
  recall    = recall_init              (from recall pipeline)
  f1        = 2 * P * R / (P + R)      (0 if both 0)

If the recall CSV is not yet available (run still in progress on the server),
the partial recall numbers can be supplied inline via --inline-recall (see
__main__ for the hardcoded fallback covering the 20 rows already logged).

Output:
  results/600q/rewriting_chains_musique_600q_openfactscore_f1_<qid>.csv
    qid, group, instruction_type, run, step, instruction_used,
    precision, recall, f1, n_facts_pred, n_supported_pred,
    n_e0_facts, n_recalled
"""

import argparse
import io
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PRECISION_CSV = (
    REPO_ROOT / "results" / "600q" / "rewriting_chains_musique_600q_openfactscore.csv"
)
DEFAULT_RECALL_CSV = (
    REPO_ROOT / "results" / "600q" / "rewriting_chains_musique_600q_openfactscore_recall.csv"
)
DEFAULT_QID = "2hop__14092_8311"

JOIN_KEYS = ["qid", "group", "instruction_type", "run", "step"]

# Partial recall data already produced on the server (rows 1-20 of the live log).
# Used only as fallback when the recall CSV is not yet on disk locally.
INLINE_RECALL_TSV = """\
qid\tgroup\tinstruction_type\trun\tstep\tn_e0_facts\tn_recalled\trecall_init
2hop__14092_8311\tcontent\telaborate\t0\t1\t217\t144\t0.664
2hop__14092_8311\tcontent\telaborate\t0\t2\t217\t123\t0.567
2hop__14092_8311\tcontent\telaborate\t0\t3\t217\t48\t0.221
2hop__14092_8311\tcontent\telaborate\t1\t1\t217\t116\t0.535
2hop__14092_8311\tcontent\telaborate\t1\t2\t217\t65\t0.300
2hop__14092_8311\tcontent\telaborate\t1\t3\t217\t65\t0.300
2hop__14092_8311\tcontent\telaborate\t2\t1\t217\t122\t0.562
2hop__14092_8311\tcontent\telaborate\t2\t2\t217\t98\t0.452
2hop__14092_8311\tcontent\telaborate\t2\t3\t217\t95\t0.438
2hop__14092_8311\tcontent\tshorten\t0\t1\t217\t83\t0.382
2hop__14092_8311\tcontent\tshorten\t0\t2\t217\t84\t0.387
2hop__14092_8311\tcontent\tshorten\t0\t3\t217\t60\t0.276
2hop__14092_8311\tcontent\tshorten\t1\t1\t217\t59\t0.272
2hop__14092_8311\tcontent\tshorten\t1\t2\t217\t54\t0.249
2hop__14092_8311\tcontent\tshorten\t1\t3\t217\t50\t0.230
2hop__14092_8311\tcontent\tshorten\t2\t1\t217\t136\t0.627
2hop__14092_8311\tcontent\tshorten\t2\t2\t217\t70\t0.323
2hop__14092_8311\tcontent\tshorten\t2\t3\t217\t65\t0.300
2hop__14092_8311\tstyle\tformality\t0\t1\t217\t150\t0.691
2hop__14092_8311\tstyle\tformality\t0\t2\t217\t127\t0.585
"""


def f1(p, r):
    if p is None or r is None:
        return None
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser(description="Compute F1 from forward OFS precision and recall CSVs.")
    ap.add_argument("--precision-csv", type=Path, default=DEFAULT_PRECISION_CSV)
    ap.add_argument("--recall-csv", type=Path, default=DEFAULT_RECALL_CSV)
    ap.add_argument("--qid", default=DEFAULT_QID)
    ap.add_argument("--use-inline", action="store_true",
                    help="Force using the inline recall snapshot instead of reading --recall-csv.")
    args = ap.parse_args()

    if not args.precision_csv.exists():
        raise FileNotFoundError(args.precision_csv)

    prec = pd.read_csv(args.precision_csv)
    prec = prec[prec["qid"] == args.qid].copy()
    prec = prec.rename(columns={
        "factscore": "precision",
        "n_facts": "n_facts_pred",
        "n_supported": "n_supported_pred",
    })

    if args.use_inline or not args.recall_csv.exists():
        if not args.use_inline:
            print(f"[info] Recall CSV not found at {args.recall_csv}; "
                  f"falling back to the inline 20-row snapshot.")
        rec = pd.read_csv(io.StringIO(INLINE_RECALL_TSV), sep="\t")
    else:
        rec = pd.read_csv(args.recall_csv)
        rec = rec[rec["qid"] == args.qid].copy()
        if "recall_init" not in rec.columns:
            raise ValueError(f"recall_init column missing in {args.recall_csv}")

    rec = rec[JOIN_KEYS + ["n_e0_facts", "n_recalled", "recall_init"]]
    rec = rec.rename(columns={"recall_init": "recall"})

    merged = prec.merge(rec, on=JOIN_KEYS, how="inner")
    if merged.empty:
        raise SystemExit(f"No overlap between precision and recall for qid={args.qid}")

    merged["f1"] = merged.apply(lambda r: f1(r["precision"], r["recall"]), axis=1)

    cols = JOIN_KEYS + [
        "instruction_used",
        "precision", "recall", "f1",
        "n_facts_pred", "n_supported_pred",
        "n_e0_facts", "n_recalled",
    ]
    merged = merged[cols].sort_values(["instruction_type", "run", "step"]).reset_index(drop=True)

    out_path = args.precision_csv.with_name(
        f"rewriting_chains_musique_600q_openfactscore_f1_{args.qid}.csv"
    )
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(merged)} rows)")
    print()

    print("=" * 90)
    print(f"OFS F1 — qid={args.qid}  ({len(merged)} chain×step rows)")
    print("=" * 90)
    show = merged[["instruction_type", "run", "step", "precision", "recall", "f1"]].copy()
    for c in ("precision", "recall", "f1"):
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print()

    print("--- mean P / R / F1 per (instruction_type, step) ---")
    agg = merged.groupby(["instruction_type", "step"])[["precision", "recall", "f1"]].mean().round(3)
    print(agg)
    print()

    print("--- mean P / R / F1 per instruction_type (across step & run) ---")
    print(merged.groupby("instruction_type")[["precision", "recall", "f1"]].mean().round(3))


if __name__ == "__main__":
    main()
