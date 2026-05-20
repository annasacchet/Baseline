#!/bin/bash
# ============================================================================
# NewsQA + FictionalQA Parallel Pipeline on H100 96GB (OLMo-3.1, 4-bit)
# ============================================================================
#
# This script orchestrates both datasets in parallel:
#   - GPU 0: NewsQA rewriting + QA
#   - GPU 1: FictionalQA rewriting + QA
#   - Then both: OpenFActScore (AFG/AFV)
#
# Estimated time: ~4.5 hours total (limited by NewsQA, the larger dataset)
# Estimated cost (H100 @ $1.5/hr): ~$6.75
#
# Usage: bash run_both_parallel_h100.sh [n_questions] [n_fictions]
# Default: 300 questions each
#

set -e

# Configuration
N_QUESTIONS="${1:-300}"
N_FICTIONS="${2:-300}"
HF_HOME="${HF_HOME:-/tmp/hf_cache}"
NEWSQA_DATASET="${NEWSQA_DATASET:-/data/combined-newsqa-data-v1.csv}"

export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export NEWSQA_DATASET="$NEWSQA_DATASET"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$REPO_ROOT"

source ~/.bashrc
conda activate baseline 2>/dev/null || conda activate base

echo "=========================================="
echo "  NewsQA + FictionalQA | H100 96GB | 4-bit"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  NewsQA questions:    $N_QUESTIONS"
echo "  FictionalQA fictions: $N_FICTIONS"
echo "  HF_HOME:             $HF_HOME"
echo "  NEWSQA_DATASET:      $NEWSQA_DATASET"
echo ""
echo "Expected time: ~4.5 hours"
echo "Expected cost (H100 @ \$1.5/hr): ~\$6.75"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# STAGE 1: Rewriting (parallelo, GPU 0 + GPU 1)
# ─────────────────────────────────────────────────────────────────────────
echo "[1/3] === REWRITING (NewsQA on GPU 0, FictionalQA on GPU 1) ==="
echo ""

NEWSQA_CHAINS="results/newsqa/rewriting_chains_newsqa_${N_QUESTIONS}.csv"
FICTIONALQA_CHAINS="results/fictionalqa/rewriting_chains_fictionalqa_${N_FICTIONS}.csv"

mkdir -p results/newsqa results/fictionalqa

CUDA_VISIBLE_DEVICES=0 python scripts/newsqa/rewriting_pipeline_newsqa.py \
  --model allenai/OLMo-3.1-32B-Instruct \
  --dataset "$NEWSQA_DATASET" \
  --output "$NEWSQA_CHAINS" \
  --n-questions "$N_QUESTIONS" \
  --n-iterations 3 \
  --temperature 0.7 \
  --max-new-tokens 4096 \
  --seed 42 \
  --use-4bit &
PID_NEWSQA=$!

CUDA_VISIBLE_DEVICES=1 python scripts/fictionalqa/rewriting_pipeline_fictionalqa.py \
  --model allenai/OLMo-3.1-32B-Instruct \
  --output "$FICTIONALQA_CHAINS" \
  --n-fictions "$N_FICTIONS" \
  --n-iterations 3 \
  --temperature 0.7 \
  --max-new-tokens 4096 \
  --seed 42 \
  --use-4bit &
PID_FICTIONALQA=$!

echo "  PID $PID_NEWSQA (NewsQA on GPU 0)"
echo "  PID $PID_FICTIONALQA (FictionalQA on GPU 1)"
echo ""
echo "  Waiting for rewriting to complete..."
wait $PID_NEWSQA $PID_FICTIONALQA
echo "✓ Rewriting complete"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# STAGE 2: Answer F1 (parallelo, GPU 0 + GPU 1)
# ─────────────────────────────────────────────────────────────────────────
echo "[2/3] === ANSWER F1 (NewsQA on GPU 0, FictionalQA on GPU 1) ==="
echo ""

NEWSQA_F1="results/newsqa/rewriting_chains_newsqa_${N_QUESTIONS}_answer_f1.csv"
FICTIONALQA_F1="results/fictionalqa/rewriting_chains_fictionalqa_${N_FICTIONS}_answer_f1.csv"

CUDA_VISIBLE_DEVICES=0 python scripts/newsqa/answer_f1_eval_newsqa.py \
  --input "$NEWSQA_CHAINS" \
  --output "$NEWSQA_F1" \
  --model allenai/OLMo-3.1-32B-Instruct \
  --batch-size 12 \
  --max-new-tokens 96 \
  --use-4bit &
PID_NEWSQA=$!

CUDA_VISIBLE_DEVICES=1 python scripts/fictionalqa/answer_f1_eval_fictionalqa.py \
  --input "$FICTIONALQA_CHAINS" \
  --output "$FICTIONALQA_F1" \
  --model allenai/OLMo-3.1-32B-Instruct \
  --batch-size 12 \
  --max-new-tokens 96 \
  --use-4bit &
PID_FICTIONALQA=$!

echo "  PID $PID_NEWSQA (NewsQA on GPU 0)"
echo "  PID $PID_FICTIONALQA (FictionalQA on GPU 1)"
echo ""
echo "  Waiting for Answer F1 to complete..."
wait $PID_NEWSQA $PID_FICTIONALQA
echo "✓ Answer F1 complete"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# STAGE 3: OpenFActScore (parallelo, GPU 0 + GPU 1)
# ─────────────────────────────────────────────────────────────────────────
echo "[3/3] === OPENFACTSCORE (NewsQA on GPU 0, FictionalQA on GPU 1) ==="
echo ""

NEWSQA_OFS="results/newsqa/rewriting_chains_newsqa_${N_QUESTIONS}_openfactscore.csv"
FICTIONALQA_OFS="results/fictionalqa/rewriting_chains_fictionalqa_${N_FICTIONS}_openfactscore.csv"

CUDA_VISIBLE_DEVICES=0 python scripts/newsqa/openfactscore_eval_newsqa.py \
  --input "$NEWSQA_CHAINS" \
  --use-4bit 2>&1 | head -20 &
PID_NEWSQA=$!

CUDA_VISIBLE_DEVICES=1 python scripts/fictionalqa/openfactscore_eval_fictionalqa.py \
  --input "$FICTIONALQA_CHAINS" \
  --use-4bit 2>&1 | head -20 &
PID_FICTIONALQA=$!

echo "  PID $PID_NEWSQA (NewsQA on GPU 0)"
echo "  PID $PID_FICTIONALQA (FictionalQA on GPU 1)"
echo ""
echo "  Waiting for OpenFActScore to complete..."
wait $PID_NEWSQA $PID_FICTIONALQA 2>/dev/null || true
echo "✓ OpenFActScore complete"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────
echo "=========================================="
echo "✓ PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "NewsQA Outputs:"
echo "  - $NEWSQA_CHAINS"
echo "  - $NEWSQA_F1"
echo "  - ${NEWSQA_CHAINS%.*}_openfactscore.csv"
echo ""
echo "FictionalQA Outputs:"
echo "  - $FICTIONALQA_CHAINS"
echo "  - $FICTIONALQA_F1"
echo "  - ${FICTIONALQA_CHAINS%.*}_openfactscore.csv"
echo ""
echo "Next: Review results and compare metrics across datasets!"
