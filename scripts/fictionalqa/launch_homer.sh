#!/bin/bash
# FictionalQA full pipeline (100 fictions) on Homer
#   GPU 0: RTX 6000 Ada (48GB)
#   GPU 1: RTX A6000     (48GB)
#
# Pipeline (sequential, all resumable):
#   1. Rewriting       (OLMo-3.1-32B-Instruct, 4-bit)
#   2. Answer F1       (OLMo-3.1-32B-Instruct, 4-bit, QA-as-grader)
#   3. OpenFActScore   (AFG OLMo-2-7B-SFT + AFV gemma-3-4b-it, 4-bit)
#   4. OFS Recall      (reuses E_0 facts from OFS details)
#   5. BLEURT          (BLEURT-20)
#   6. BERTScore       (roberta-large layer 17)
#   7. Perplexity      (OLMo-3.1-32B-Instruct, sliding-window, 4-bit)
#
# Length per-row is already in the chains CSV (n_tokens). No separate step.
#
# Usage:
#   tmux new -s fictional100
#   bash scripts/fictionalqa/launch_homer.sh

set -e

export HF_HOME="/mnt/dmif-nas/mitel/sacchet/hf_cache"
export HF_HUB_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache/hub"
export TRANSFORMERS_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache"

CHAINS="results/fictionalqa/rewriting_chains_fictionalqa.csv"
F1_OUT="results/fictionalqa/rewriting_chains_fictionalqa_answer_f1.csv"
OFS_OUT="results/fictionalqa/rewriting_chains_fictionalqa_openfactscore.csv"
OFS_DETAILS="results/fictionalqa/rewriting_chains_fictionalqa_openfactscore_details.csv"
BLEURT_OUT="results/fictionalqa/rewriting_chains_fictionalqa_bleurt.csv"
BERT_OUT="results/fictionalqa/rewriting_chains_fictionalqa_bertscore.csv"

cd ~/Baseline
mkdir -p results/fictionalqa
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate baseline 2>/dev/null || conda activate base 2>/dev/null || true

echo "=========================================="
echo "FictionalQA full pipeline — Homer (100q)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""

# --- 1. Rewriting ---
echo ""
echo "[1/7] Rewriting (100 fictions, 4-bit)..."
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
echo "[2/7] Answer F1..."
python3 scripts/fictionalqa/answer_f1_eval_fictionalqa.py \
  --input "$CHAINS" \
  --output "$F1_OUT" \
  --batch-size 8 \
  --use-4bit

# --- 3. OpenFActScore (forward; produces details with label='E0') ---
echo ""
echo "[3/7] OpenFActScore (forward)..."
python3 scripts/fictionalqa/openfactscore_eval_fictionalqa.py \
  --input "$CHAINS" \
  --output "$OFS_OUT" \
  --use-4bit

# --- 4. OFS Recall (reuses E_0 facts from step 3 details) ---
echo ""
echo "[4/7] OpenFActScore RECALL..."
python3 scripts/fictionalqa/openfactscore_recall_fictionalqa.py \
  --chain-csv "$CHAINS" \
  --details-csv "$OFS_DETAILS" \
  --use-4bit

# --- 5. BLEURT ---
echo ""
echo "[5/7] BLEURT..."
python3 scripts/fictionalqa/bleurt_eval_fictionalqa.py \
  --input "$CHAINS" \
  --f1-csv "$F1_OUT" \
  --output "$BLEURT_OUT" \
  --batch-size 64

# --- 6. BERTScore ---
echo ""
echo "[6/7] BERTScore..."
python3 scripts/fictionalqa/bertscore_eval_fictionalqa.py \
  --input "$CHAINS" \
  --output "$BERT_OUT" \
  --batch-size 32

# --- 7. Perplexity ---
echo ""
echo "[7/7] Perplexity..."
python3 scripts/_common/perplexity_eval.py \
  --dataset fictionalqa \
  --use-4bit \
  --save-stats

echo ""
echo "=========================================="
echo "FictionalQA pipeline complete."
echo "  Chains:      $CHAINS  (n_tokens column = per-row length)"
echo "  Answer F1:   $F1_OUT"
echo "  OFS:         $OFS_OUT"
echo "  OFS Recall:  results/fictionalqa/rewriting_chains_fictionalqa_openfactscore_recall.csv"
echo "  BLEURT:      $BLEURT_OUT"
echo "  BERTScore:   $BERT_OUT"
echo "  Perplexity:  results/fictionalqa/rewriting_chains_fictionalqa_perplexity.csv"
echo "=========================================="
