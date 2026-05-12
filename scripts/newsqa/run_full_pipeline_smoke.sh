#!/bin/bash
# Full NewsQA pipeline smoke test: rewriting → Answer F1 → OpenFActScore
# Run on Lisa or Homer depending on setup

set -e

# Configuration
SERVER="${SERVER:-lisa}"  # lisa or homer
NEWSQA_DATASET="${NEWSQA_DATASET:-/mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv}"
HF_HOME="${HF_HOME:-/mnt/dmif-nas/mitel/sacchet/hf_cache}"
USE_4BIT="${USE_4BIT:-true}"  # true for Lisa (4-bit), false for Homer (bf16)

# Export env vars
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export NEWSQA_DATASET="$NEWSQA_DATASET"

# Paths
REPO_ROOT=$(cd "$(dirname "$0")/../../" && pwd)
RESULTS_DIR="$REPO_ROOT/results/newsqa"
mkdir -p "$RESULTS_DIR"

CHAINS_CSV="$RESULTS_DIR/rewriting_chains_newsqa_smoke.csv"
ANSWER_F1_CSV="$RESULTS_DIR/rewriting_chains_newsqa_smoke_answer_f1.csv"
OFS_CSV="$RESULTS_DIR/rewriting_chains_newsqa_smoke_openfactscore.csv"

cd "$REPO_ROOT"

# Activate conda
source ~/.bashrc
conda activate baseline 2>/dev/null || conda activate base

echo "=========================================="
echo "NewsQA Full Pipeline Smoke Test"
echo "=========================================="
echo "Server: $SERVER"
echo "4-bit: $USE_4BIT"
echo "HF_HOME: $HF_HOME"
echo "Dataset: $NEWSQA_DATASET"
echo ""

# -----------
# Stage 1: Rewriting
# -----------
echo "[1/3] Rewriting pipeline (1 story, all 4 instructions x 3 wordings)"
echo ""

4bit_flag=""
if [ "$USE_4BIT" = "true" ]; then
    4bit_flag="--use-4bit"
fi

python scripts/newsqa/rewriting_pipeline_newsqa.py \
  --model allenai/OLMo-3.1-32B-Instruct \
  --dataset "$NEWSQA_DATASET" \
  --output "$CHAINS_CSV" \
  --n-questions 1 \
  --n-iterations 3 \
  --temperature 0.7 \
  --max-new-tokens 4096 \
  --seed 42 \
  --smoke-test \
  $4bit_flag

echo "✓ Stage 1 complete: $CHAINS_CSV"
echo ""

# -----------
# Stage 2: Answer F1
# -----------
echo "[2/3] Answer F1 evaluation"
echo ""

python scripts/newsqa/answer_f1_eval_newsqa.py \
  --input "$CHAINS_CSV" \
  --output "$ANSWER_F1_CSV" \
  --model allenai/OLMo-3.1-32B-Instruct \
  --batch-size 4 \
  --max-new-tokens 96 \
  --smoke-test \
  $4bit_flag

echo "✓ Stage 2 complete: $ANSWER_F1_CSV"
echo ""

# -----------
# Stage 3: OpenFActScore (only if demos.json exists)
# -----------
DEMOS="$REPO_ROOT/data/demons.json"
if [ -f "$DEMOS" ]; then
    echo "[3/3] OpenFActScore evaluation"
    echo ""
    python scripts/newsqa/openfactscore_eval_newsqa.py \
      --input "$CHAINS_CSV" \
      --demos "$DEMOS" \
      --afg-model allenai/OLMo-2-1124-7B-SFT \
      --afv-model google/gemma-3-4b-it \
      --limit 12 \
      $4bit_flag
    echo "✓ Stage 3 complete: $OFS_CSV"
else
    echo "[3/3] Skipping OpenFActScore (demons.json not found at $DEMOS)"
fi

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo "Outputs:"
echo "  Chains:      $CHAINS_CSV"
echo "  Answer F1:   $ANSWER_F1_CSV"
if [ -f "$DEMOS" ]; then
    echo "  OpenFActScore: $OFS_CSV"
fi
