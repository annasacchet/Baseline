#!/bin/bash
# NewsQA eval-only pipeline on Lisa — runs F1 (from scratch), OFS (resume), BLEURT (resume)
# Rewriting is already done in rewriting_chains_newsqa_100q.csv
#
# Usage:
#   tmux new -s newsqa_eval
#   bash scripts/newsqa/launch_eval_lisa.sh

set -e

export HF_HOME="/mnt/dmif-nas/mitel/sacchet/hf_cache"
export HF_HUB_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache/hub"
export TRANSFORMERS_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache"

CHAINS="results/newsqa/rewriting_chains_newsqa_100q.csv"
F1_OUT="results/newsqa/rewriting_chains_newsqa_100q_answer_f1_span.csv"
OFS_OUT="results/newsqa/rewriting_chains_newsqa_100q_openfactscore.csv"
BLEURT_OUT="results/newsqa/rewriting_chains_newsqa_100q_bleurt.csv"

cd ~/Baseline
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate baseline 2>/dev/null || conda activate base 2>/dev/null || true

echo "=========================================="
echo "NewsQA eval pipeline — Lisa"
echo "Chains: $CHAINS"
echo "=========================================="

# --- 1. Answer F1 (from scratch — overwrite) ---
echo ""
echo "[1/3] Answer F1 (overwriting $F1_OUT)..."
rm -f "$F1_OUT"
python3 scripts/newsqa/answer_f1_eval_newsqa.py \
  --input "$CHAINS" \
  --output "$F1_OUT" \
  --batch-size 8 \
  --use-4bit

# --- 2. OpenFActScore (resume from where it stopped) ---
echo ""
echo "[2/3] OpenFActScore (resuming $OFS_OUT)..."
python3 scripts/newsqa/openfactscore_eval_newsqa.py \
  --input "$CHAINS" \
  --output "$OFS_OUT" \
  --use-4bit

# --- 3. BLEURT (resume) ---
echo ""
echo "[3/3] BLEURT (resuming $BLEURT_OUT)..."
python3 scripts/newsqa/bleurt_eval_newsqa.py \
  --input "$CHAINS" \
  --f1-csv "$F1_OUT" \
  --output "$BLEURT_OUT" \
  --batch-size 64

echo ""
echo "=========================================="
echo "NewsQA eval complete."
echo "  F1:     $F1_OUT"
echo "  OFS:    $OFS_OUT"
echo "  BLEURT: $BLEURT_OUT"
echo "=========================================="
