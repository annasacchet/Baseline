#!/bin/bash
# FictionalQA full pipeline on Lisa — rewriting (100 docs) + F1 + OFS + BLEURT
#
# Usage:
#   tmux new -s fictionalqa
#   bash scripts/fictionalqa/launch_full_lisa.sh

set -e

export HF_HOME="/mnt/dmif-nas/mitel/sacchet/hf_cache"
export HF_HUB_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache/hub"
export TRANSFORMERS_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache"

CHAINS="results/fictionalqa/rewriting_chains_fictionalqa.csv"
F1_OUT="results/fictionalqa/rewriting_chains_fictionalqa_answer_f1.csv"
OFS_OUT="results/fictionalqa/rewriting_chains_fictionalqa_openfactscore.csv"
BLEURT_OUT="results/fictionalqa/rewriting_chains_fictionalqa_bleurt.csv"

cd ~/Baseline
source ~/.bashrc
conda activate baseline 2>/dev/null || conda activate base

echo "=========================================="
echo "FictionalQA full pipeline — Lisa"
echo "=========================================="

# --- 1. Rewriting (100 fictions, resume if partially done) ---
echo ""
echo "[1/4] Rewriting pipeline (100 fictions)..."
python3 scripts/fictionalqa/rewriting_pipeline_fictionalqa.py \
  --model allenai/OLMo-3.1-32B-Instruct \
  --output "$CHAINS" \
  --n-fictions 100 \
  --n-iterations 3 \
  --temperature 0.7 \
  --max-new-tokens 4096 \
  --seed 42 \
  --use-4bit

# --- 2. Answer F1 ---
echo ""
echo "[2/4] Answer F1..."
python3 scripts/fictionalqa/answer_f1_eval_fictionalqa.py \
  --input "$CHAINS" \
  --output "$F1_OUT" \
  --batch-size 8 \
  --use-4bit

# --- 3. OpenFActScore ---
echo ""
echo "[3/4] OpenFActScore..."
python3 scripts/fictionalqa/openfactscore_eval_fictionalqa.py \
  --input "$CHAINS" \
  --output "$OFS_OUT" \
  --use-4bit

# --- 4. BLEURT ---
echo ""
echo "[4/4] BLEURT..."
python3 scripts/fictionalqa/bleurt_eval_fictionalqa.py \
  --input "$CHAINS" \
  --f1-csv "$F1_OUT" \
  --output "$BLEURT_OUT" \
  --batch-size 64

echo ""
echo "=========================================="
echo "FictionalQA pipeline complete."
echo "  Chains:  $CHAINS"
echo "  F1:      $F1_OUT"
echo "  OFS:     $OFS_OUT"
echo "  BLEURT:  $BLEURT_OUT"
echo "=========================================="
