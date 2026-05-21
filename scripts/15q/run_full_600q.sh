#!/usr/bin/env bash
# Full MuSiQue rewriting pipeline + all evals on 600 questions (200 per hop).
#
# Stages (each one is skipped if its output already exists, unless --force):
#   1) Rewriting     -> results/rewriting_chains_musique_600q.csv
#   2) Token length  -> already in the rewriting CSV (column `n_tokens`, computed
#                       with the rewriter's own tokenizer). No separate stage.
#   3) Answer F1     -> results/rewriting_chains_musique_600q_answer_f1.csv
#   4) BERTScore     -> results/rewriting_chains_musique_600q_bertscore.csv
#   5) BLEURT        -> results/rewriting_chains_musique_600q_bleurt.csv
#   6) OpenFActScore -> results/rewriting_chains_musique_600q_openfactscore.csv
#                       (full: every step>0 row)
#
# Usage (on Lisa, inside tmux):
#   bash scripts/15q/run_full_600q.sh           # normal run, resumes if interrupted
#   FORCE=1 bash scripts/15q/run_full_600q.sh   # rerun everything from scratch
#   SKIP_OFS=1 bash scripts/15q/run_full_600q.sh  # skip OpenFActScore (longest stage)
#
# What it prints:
#   - Per-stage header with wall-clock start time.
#   - Live stdout of the underlying python script (so you see internal progress bars).
#   - Per-stage duration and a running total at the end of each stage.
#   - Final summary with output paths + size.
#
# Layout assumption: the script is in scripts/15q/ and is run from anywhere; it
# resolves the repo root from its own location.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
LOGS_DIR="$REPO_ROOT/logs"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

DATASET="$REPO_ROOT/musique_ans_v1.0_dev.jsonl"

MODEL="${MODEL:-allenai/OLMo-3.1-32B-Instruct}"
QA_MODEL="${QA_MODEL:-$MODEL}"
N_PER_HOP="${N_PER_HOP:-200}"           # 200 * 3 hops = 600 questions
N_ITERATIONS="${N_ITERATIONS:-3}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
SEED="${SEED:-42}"

# Per Lisa (2x3090). For Homer/H100 in bf16, run with USE_4BIT=0.
USE_4BIT="${USE_4BIT:-1}"
FOURBIT_FLAG=""
[ "$USE_4BIT" = "1" ] && FOURBIT_FLAG="--use-4bit"

# HF cache on NAS (mandatory on Lisa).
export HF_HOME="${HF_HOME:-/mnt/dmif-nas/mitel/sacchet/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

# Conda activation (fresh tmux sessions don't inherit it).
CONDA_ENV="${CONDA_ENV:-baseline}"
if ! command -v python >/dev/null 2>&1 || [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    # Try the standard conda hook locations.
    for c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh" \
             "/opt/conda/etc/profile.d/conda.sh"; do
        if [ -f "$c" ]; then
            # shellcheck disable=SC1090
            source "$c"
            break
        fi
    done
    conda activate "$CONDA_ENV" 2>/dev/null || conda activate base 2>/dev/null || true
fi
if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: 'python' not found even after attempting conda activation." >&2
    echo "  Activate your env manually then re-run, e.g.:" >&2
    echo "    source ~/.bashrc && conda activate baseline && bash $0" >&2
    exit 1
fi

# HF token (optional but recommended — kills the unauthenticated-request warning).
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARN: HF_TOKEN not set. Public models still work; gated models will fail." >&2
    echo "      Export HF_TOKEN before running (e.g. add it to ~/.bashrc)." >&2
fi

FORCE="${FORCE:-0}"
SKIP_OFS="${SKIP_OFS:-0}"
SKIP_BLEURT="${SKIP_BLEURT:-0}"
SKIP_BERTSCORE="${SKIP_BERTSCORE:-0}"
SKIP_F1="${SKIP_F1:-0}"

# Output paths
TAG="musique_600q"
CHAINS_CSV="$RESULTS_DIR/rewriting_chains_${TAG}.csv"
F1_CSV="$RESULTS_DIR/rewriting_chains_${TAG}_answer_f1.csv"
BERTSCORE_CSV="$RESULTS_DIR/rewriting_chains_${TAG}_bertscore.csv"
BLEURT_CSV="$RESULTS_DIR/rewriting_chains_${TAG}_bleurt.csv"
OFS_CSV="$RESULTS_DIR/rewriting_chains_${TAG}_openfactscore.csv"

DEMOS="$REPO_ROOT/data/demons.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RUN_START=$(date +%s)
STAGE_LOG="$LOGS_DIR/run_full_${TAG}_$(date +%Y%m%d_%H%M%S).log"

fmt_secs() {
    # Convert seconds -> Hh Mm Ss
    local s=$1
    printf "%dh %02dm %02ds" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

stage_header() {
    local n=$1 ; local title=$2 ; local out=$3
    echo ""
    echo "=============================================================="
    echo "[$n] $title"
    echo "      start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "      output: $out"
    local elapsed=$(( $(date +%s) - RUN_START ))
    echo "      elapsed so far: $(fmt_secs $elapsed)"
    echo "=============================================================="
}

stage_footer() {
    local n=$1 ; local stage_start=$2 ; local out=$3
    local now=$(date +%s)
    local dur=$((now - stage_start))
    local total=$((now - RUN_START))
    local size="-"
    [ -f "$out" ] && size=$(du -h "$out" | cut -f1)
    echo ""
    echo "  ✓ stage [$n] done in $(fmt_secs $dur) | total: $(fmt_secs $total) | output: $out ($size)"
}

skip_or_run() {
    # $1=name, $2=output, $3...=command
    local name=$1 ; local out=$2 ; shift 2
    if [ -f "$out" ] && [ "$FORCE" != "1" ]; then
        echo "  ↷ $name: output already exists ($out) — skipping (use FORCE=1 to rerun)."
        return 0
    fi
    "$@"
}

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
{
echo "=============================================================="
echo " Full MuSiQue 600q pipeline"
echo "=============================================================="
echo " repo:            $REPO_ROOT"
echo " dataset:         $DATASET"
echo " model:           $MODEL"
echo " qa model:        $QA_MODEL"
echo " n-per-hop:       $N_PER_HOP  (total: $((N_PER_HOP*3)) questions)"
echo " n-iterations:    $N_ITERATIONS"
echo " temperature:     $TEMPERATURE"
echo " max-new-tokens:  $MAX_NEW_TOKENS"
echo " seed:            $SEED"
echo " use-4bit:        $USE_4BIT"
echo " HF_HOME:         $HF_HOME"
echo " log file:        $STAGE_LOG"
echo " force rerun:     $FORCE"
echo " skip OFS:        $SKIP_OFS"
echo " skip BLEURT:     $SKIP_BLEURT"
echo " skip BERTScore:  $SKIP_BERTSCORE"
echo " skip Answer F1:  $SKIP_F1"
echo "--------------------------------------------------------------"
echo " stages will run sequentially. Each stage prints its own"
echo " progress; ETA between stages is computed by the wrapper."
echo "=============================================================="
} | tee "$STAGE_LOG"

# ---------------------------------------------------------------------------
# [1/5] Rewriting
# ---------------------------------------------------------------------------
{
    stage_header "1/5" "Rewriting (600 questions, $N_ITERATIONS iterations each)" "$CHAINS_CSV"
    t0=$(date +%s)
    skip_or_run "Rewriting" "$CHAINS_CSV" \
        python scripts/15q/rewriting_pipeline.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --output "$CHAINS_CSV" \
            --n-per-hop "$N_PER_HOP" \
            --n-iterations "$N_ITERATIONS" \
            --temperature "$TEMPERATURE" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --seed "$SEED" \
            $FOURBIT_FLAG
    stage_footer "1/5" "$t0" "$CHAINS_CSV"
} 2>&1 | tee -a "$STAGE_LOG"

if [ ! -f "$CHAINS_CSV" ]; then
    echo "ERROR: rewriting CSV not produced — aborting." | tee -a "$STAGE_LOG"
    exit 1
fi

# ---------------------------------------------------------------------------
# [2/5] Answer F1
# ---------------------------------------------------------------------------
if [ "$SKIP_F1" = "1" ]; then
    echo "  ↷ Skipping Answer F1 (SKIP_F1=1)" | tee -a "$STAGE_LOG"
else
{
    stage_header "2/5" "Answer F1 (QA model: $QA_MODEL)" "$F1_CSV"
    t0=$(date +%s)
    skip_or_run "Answer F1" "$F1_CSV" \
        python scripts/15q/answer_f1_eval.py \
            --input "$CHAINS_CSV" \
            --output "$F1_CSV" \
            --dataset "$DATASET" \
            --model "$QA_MODEL" \
            --batch-size 8 \
            $FOURBIT_FLAG
    stage_footer "2/5" "$t0" "$F1_CSV"
} 2>&1 | tee -a "$STAGE_LOG"
fi

# ---------------------------------------------------------------------------
# [3/5] BERTScore (baseline E_t vs E_0, consecutive E_t vs E_{t-1})
# ---------------------------------------------------------------------------
if [ "$SKIP_BERTSCORE" = "1" ]; then
    echo "  ↷ Skipping BERTScore (SKIP_BERTSCORE=1)" | tee -a "$STAGE_LOG"
else
{
    stage_header "3/5" "BERTScore (baseline + consecutive)" "$BERTSCORE_CSV"
    t0=$(date +%s)
    skip_or_run "BERTScore" "$BERTSCORE_CSV" \
        python scripts/15q/bertscore_eval.py \
            --input "$CHAINS_CSV" \
            --output "$BERTSCORE_CSV" \
            --batch-size 16
    stage_footer "3/5" "$t0" "$BERTSCORE_CSV"
} 2>&1 | tee -a "$STAGE_LOG"
fi

# ---------------------------------------------------------------------------
# [4/5] BLEURT (baseline + consecutive + answer-level if F1 CSV exists)
# ---------------------------------------------------------------------------
if [ "$SKIP_BLEURT" = "1" ]; then
    echo "  ↷ Skipping BLEURT (SKIP_BLEURT=1)" | tee -a "$STAGE_LOG"
else
{
    stage_header "4/5" "BLEURT (baseline + consecutive + answer-level)" "$BLEURT_CSV"
    t0=$(date +%s)
    F1_ARG=()
    if [ -f "$F1_CSV" ]; then
        F1_ARG=(--f1-csv "$F1_CSV")
    else
        echo "  note: no Answer F1 CSV at $F1_CSV — answer-level BLEURT will be skipped."
    fi
    skip_or_run "BLEURT" "$BLEURT_CSV" \
        python scripts/300q/bleurt_eval_300q.py \
            --input "$CHAINS_CSV" \
            --output "$BLEURT_CSV" \
            "${F1_ARG[@]}" \
            --batch-size 64
    stage_footer "4/5" "$t0" "$BLEURT_CSV"
} 2>&1 | tee -a "$STAGE_LOG"
fi

# ---------------------------------------------------------------------------
# [5/5] OpenFActScore (full)
# ---------------------------------------------------------------------------
if [ "$SKIP_OFS" = "1" ]; then
    echo "  ↷ Skipping OpenFActScore (SKIP_OFS=1)" | tee -a "$STAGE_LOG"
else
    if [ ! -f "$DEMOS" ]; then
        echo "  ↷ Skipping OpenFActScore: demos file not found at $DEMOS" | tee -a "$STAGE_LOG"
    else
{
        stage_header "5/5" "OpenFActScore (FULL — every step>0 row)" "$OFS_CSV"
        t0=$(date +%s)
        skip_or_run "OpenFActScore" "$OFS_CSV" \
            python scripts/15q/openfactscore_eval.py \
                --input "$CHAINS_CSV" \
                --demos "$DEMOS" \
                $FOURBIT_FLAG
        # The script writes "<input_stem>_openfactscore.csv" next to the input.
        # If that path differs from our expected $OFS_CSV (it shouldn't — same naming
        # convention), copy it over.
        EXPECTED_OFS="${CHAINS_CSV%.csv}_openfactscore.csv"
        if [ -f "$EXPECTED_OFS" ] && [ "$EXPECTED_OFS" != "$OFS_CSV" ]; then
            cp "$EXPECTED_OFS" "$OFS_CSV"
        fi
        stage_footer "5/5" "$t0" "$OFS_CSV"
} 2>&1 | tee -a "$STAGE_LOG"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
{
total=$(( $(date +%s) - RUN_START ))
echo ""
echo "=============================================================="
echo " Pipeline complete in $(fmt_secs $total)"
echo "=============================================================="
for f in "$CHAINS_CSV" "$F1_CSV" "$BERTSCORE_CSV" "$BLEURT_CSV" "$OFS_CSV"; do
    if [ -f "$f" ]; then
        rows=$(($(wc -l < "$f") - 1))
        size=$(du -h "$f" | cut -f1)
        printf "  %-60s rows=%-6s size=%s\n" "$(basename "$f")" "$rows" "$size"
    else
        printf "  %-60s [MISSING]\n" "$(basename "$f")"
    fi
done
echo ""
echo " Full log: $STAGE_LOG"
echo "=============================================================="
} | tee -a "$STAGE_LOG"
