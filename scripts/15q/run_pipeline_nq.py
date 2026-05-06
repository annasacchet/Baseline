"""
run_pipeline_nq.py — rewriting experiment pipeline, N questions.

Runs the full sequence:
  1. rewriting_pipeline.py         — generate rewriting chains
  2. openfactscore_eval.py         — FactScore evaluation  (parallel if --parallel)
  3. bertscore_eval.py             — BERTScore evaluation  (parallel if --parallel)
  4. answer_f1_eval.py             — Answer F1 evaluation  (parallel if --parallel)
  5. visualize_trajectories_15q.py — trajectory plots
  6. visualize_answerability_15q.py — answerability plots
  7. sign_tests_15q.py             — sign tests (all metrics)
  8. factscore_sign_tests_15q.py   — FactScore sign tests

Use --tag to name the experiment (e.g. 15q, 300q) — all CSVs and plots
are namespaced under that tag so runs never overwrite each other.
Use --parallel to split eval steps across multiple GPUs (default: 2).

Usage examples:
  # 15q pilot, single GPU
  python scripts/15q/run_pipeline_nq.py --tag 15q

  # 300q, 100 questions per hop, parallel eval on 2 GPUs
  python scripts/15q/run_pipeline_nq.py --tag 300q --parallel --rewriting-args -- --n-per-hop 100

  # Skip rewriting (chains already on server), run everything else in parallel
  python scripts/15q/run_pipeline_nq.py --tag 300q --parallel --skip rewriting

  # Force re-run everything
  python scripts/15q/run_pipeline_nq.py --tag 300q --parallel --force
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS   = Path(__file__).resolve().parent

STEP_ORDER = [
    "rewriting",
    "ofs",
    "bertscore",
    "f1",
    "plots_traj",
    "plots_ans",
    "sign_tests",
    "sign_tests_ofs",
]

STEP_LABELS = {
    "rewriting":      "1. Rewriting pipeline",
    "ofs":            "2. OpenFactScore eval",
    "bertscore":      "3. BERTScore eval",
    "f1":             "4. Answer F1 eval",
    "plots_traj":     "5. Trajectory plots",
    "plots_ans":      "6. Answerability plots",
    "sign_tests":     "7. Sign tests (all metrics)",
    "sign_tests_ofs": "8. FactScore sign tests",
}

# Eval steps that support parallel sharding
PARALLELIZABLE = {"ofs", "bertscore", "f1"}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def step_scripts(tag: str) -> dict:
    res_dir = REPO_ROOT / "results" / tag
    return {
        "rewriting": (
            SCRIPTS / "rewriting_pipeline.py",
            ["--output", str(res_dir / f"rewriting_chains_{tag}.csv")],
        ),
        "ofs": (
            SCRIPTS / "openfactscore_eval.py",
            ["--input", str(res_dir / f"rewriting_chains_{tag}.csv")],
        ),
        "bertscore": (
            SCRIPTS / "bertscore_eval.py",
            ["--input",  str(res_dir / f"rewriting_chains_{tag}.csv"),
             "--output", str(res_dir / f"rewriting_chains_{tag}_bertscore.csv")],
        ),
        "f1": (
            SCRIPTS / "answer_f1_eval.py",
            ["--input", str(res_dir / f"rewriting_chains_{tag}.csv")],
        ),
        "plots_traj":     (SCRIPTS / "visualize_trajectories_15q.py",  ["--tag", tag]),
        "plots_ans":      (SCRIPTS / "visualize_answerability_15q.py", ["--tag", tag]),
        "sign_tests":     (SCRIPTS / "sign_tests_15q.py",              ["--tag", tag]),
        "sign_tests_ofs": (SCRIPTS / "factscore_sign_tests_15q.py",    ["--tag", tag]),
    }


def outputs(tag: str) -> dict:
    res_dir  = REPO_ROOT / "results" / tag
    plot_dir = REPO_ROOT / "results" / "plots" / tag
    return {
        "rewriting":      res_dir  / f"rewriting_chains_{tag}.csv",
        "ofs":            res_dir  / f"rewriting_chains_{tag}_openfactscore.csv",
        "bertscore":      res_dir  / f"rewriting_chains_{tag}_bertscore.csv",
        "f1":             res_dir  / f"rewriting_chains_{tag}_answer_f1.csv",
        "plots_traj":     plot_dir / "traj_f1_by_instruction.pdf",
        "plots_ans":      plot_dir / "answerability_f1_by_instruction.pdf",
        "sign_tests":     res_dir  / f"sign_tests_{tag}.csv",
        "sign_tests_ofs": res_dir  / f"factscore_sign_tests_{tag}.csv",
    }


# ---------------------------------------------------------------------------
# Sequential step
# ---------------------------------------------------------------------------

def run_step(name: str, script: Path, default_args: list, output: Path, force: bool) -> bool:
    if not force and output and output.exists():
        print(f"  [SKIP] {STEP_LABELS[name]} — output already exists: {output.name}")
        return True

    cmd = [sys.executable, str(script)] + default_args
    print(f"\n{'='*60}")
    print(f"  {STEP_LABELS[name]}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  [ERROR] {STEP_LABELS[name]} failed (exit {result.returncode}).")
        return False

    print(f"\n  [OK] {STEP_LABELS[name]} done.")
    return True


# ---------------------------------------------------------------------------
# Parallel eval helpers
# ---------------------------------------------------------------------------

def split_qids(chain_csv: Path, n_shards: int) -> list:
    df   = pd.read_csv(chain_csv, usecols=["qid"])
    qids = df["qid"].unique().tolist()
    shards = [[] for _ in range(n_shards)]
    for i, qid in enumerate(qids):
        shards[i % n_shards].append(qid)
    return shards


def write_shard_csv(chain_csv: Path, qids: list, idx: int) -> Path:
    df    = pd.read_csv(chain_csv)
    out   = chain_csv.with_name(f"{chain_csv.stem}_shard{idx}.csv")
    df[df["qid"].isin(qids)].to_csv(out, index=False)
    print(f"    Shard {idx}: {len(qids)} qids → {out.name}")
    return out


def shard_output_path(shard_csv: Path, eval_type: str) -> Path:
    if eval_type == "ofs":
        return shard_csv.with_name(shard_csv.stem + "_openfactscore.csv")
    if eval_type == "bertscore":
        return shard_csv.with_name(shard_csv.stem + "_bertscore.csv")
    if eval_type == "f1":
        return shard_csv.with_name(shard_csv.stem + "_answer_f1.csv")
    raise ValueError(eval_type)


def build_shard_cmd(eval_type: str, shard_csv: Path) -> list:
    shard_out = shard_output_path(shard_csv, eval_type)
    scripts_map = {
        "ofs":       SCRIPTS / "openfactscore_eval.py",
        "bertscore": SCRIPTS / "bertscore_eval.py",
        "f1":        SCRIPTS / "answer_f1_eval.py",
    }
    script = scripts_map[eval_type]
    if eval_type == "ofs":
        return [sys.executable, str(script), "--input", str(shard_csv)]
    return [sys.executable, str(script),
            "--input", str(shard_csv), "--output", str(shard_out)]


def merge_csvs(paths: list, final_out: Path):
    dfs = [pd.read_csv(p) for p in paths if p.exists()]
    if not dfs:
        print(f"  ERROR: no shard outputs to merge for {final_out.name}")
        return
    pd.concat(dfs, ignore_index=True).to_csv(final_out, index=False)
    print(f"  Merged {len(dfs)} shards → {final_out.name} ({sum(len(d) for d in dfs)} rows)")


def run_parallel(name: str, eval_type: str, chain_csv: Path,
                 final_out: Path, gpus: list, force: bool) -> bool:
    if not force and final_out.exists():
        print(f"  [SKIP] {STEP_LABELS[name]} — output already exists: {final_out.name}")
        return True

    n_shards = len(gpus)
    print(f"\n{'='*60}")
    print(f"  {STEP_LABELS[name]} [PARALLEL — {n_shards} shards, GPUs={gpus}]")
    print(f"{'='*60}")

    shards    = split_qids(chain_csv, n_shards)
    shard_csvs = [write_shard_csv(chain_csv, qids, i) for i, qids in enumerate(shards)]
    shard_outs = [shard_output_path(csv, eval_type) for csv in shard_csvs]

    procs = []
    t_start = time.time()
    for i, (shard_csv, gpu) in enumerate(zip(shard_csvs, gpus)):
        cmd      = build_shard_cmd(eval_type, shard_csv)
        env      = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = shard_csv.with_suffix(f".shard{i}.log")
        log_file = open(log_path, "w")
        print(f"  Shard {i} → GPU {gpu} | log: {log_path.name}")
        p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((i, p, log_file, log_path))

    print(f"\n  Waiting for {n_shards} processes...")
    failed_shards = []
    for i, p, log_file, log_path in procs:
        p.wait()
        log_file.close()
        elapsed = (time.time() - t_start) / 60
        if p.returncode == 0:
            print(f"  [OK]    Shard {i} — {elapsed:.1f} min elapsed")
        else:
            print(f"  [ERROR] Shard {i} failed (exit {p.returncode}) | see {log_path.name}")
            failed_shards.append(i)

    print(f"\n  Merging shard outputs...")
    merge_csvs(shard_outs, final_out)

    # OFS also produces a details CSV
    if eval_type == "ofs":
        detail_outs = [p.with_name(p.stem.replace("_openfactscore", "") + "_openfactscore_details.csv")
                       for p in shard_outs]
        final_details = chain_csv.with_name(chain_csv.stem + "_openfactscore_details.csv")
        merge_csvs(detail_outs, final_details)

    # Cleanup
    for f in shard_csvs + shard_outs:
        if f.exists():
            f.unlink()
    for _, _, _, log_path in procs:
        if log_path.exists():
            log_path.unlink()

    if failed_shards:
        print(f"  WARNING: shards {failed_shards} had errors.")
        return False

    print(f"  [OK] {STEP_LABELS[name]} done.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Master pipeline for the rewriting experiment.")
    parser.add_argument(
        "--tag", default="15q",
        help="Dataset tag — controls all input/output paths (default: 15q).",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run eval steps (ofs, bertscore, f1) in parallel across multiple GPUs.",
    )
    parser.add_argument(
        "--gpus", nargs="+", default=["0", "1"],
        help="GPU indices for parallel shards (default: 0 1). One shard per GPU.",
    )
    parser.add_argument(
        "--skip", nargs="+", default=[], choices=STEP_ORDER,
        help="Steps to skip.",
    )
    parser.add_argument(
        "--only", nargs="+", default=[], choices=STEP_ORDER,
        help="Run only these steps (overrides --skip).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run steps even if output already exists.",
    )
    parser.add_argument(
        "--rewriting-args", nargs="+", default=[],
        help="Extra args forwarded to rewriting_pipeline.py (e.g. --n-per-hop 100 --model allenai/OLMo-3.1-32B-Instruct --use-4bit).",
    )
    args = parser.parse_args()

    tag          = args.tag
    steps        = step_scripts(tag)
    out          = outputs(tag)
    steps_to_run = args.only if args.only else [s for s in STEP_ORDER if s not in args.skip]
    chain_csv    = REPO_ROOT / "results" / tag / f"rewriting_chains_{tag}.csv"

    (REPO_ROOT / "results" / tag).mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results" / "plots" / tag / "png").mkdir(parents=True, exist_ok=True)

    print(f"\nTag:      {tag}")
    print(f"Parallel: {args.parallel} (GPUs: {args.gpus})")
    print(f"Steps:    {steps_to_run}")

    t_pipeline = time.time()

    for name in STEP_ORDER:
        if name not in steps_to_run:
            continue

        if args.parallel and name in PARALLELIZABLE:
            eval_type = {"ofs": "ofs", "bertscore": "bertscore", "f1": "f1"}[name]
            ok = run_parallel(name, eval_type, chain_csv, out[name], args.gpus, args.force)
        else:
            script, default_args = steps[name]
            extra = args.rewriting_args if name == "rewriting" else []
            ok = run_step(name, script, default_args + extra, out.get(name), args.force)

        if not ok:
            print(f"\n  Stopping pipeline due to failure in: {name}")
            sys.exit(1)

    elapsed = (time.time() - t_pipeline) / 60
    print(f"\n{'='*60}")
    print(f"  Pipeline completed. Tag: {tag} | Total time: {elapsed:.1f} min")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
