#!/bin/bash
# RQ1 baseline pipeline on FictionalQA, 90 fictions — OLMo-3.1-32B-Instruct
# loaded in 4-bit NF4 via vLLM bitsandbytes on Homer.
#
# Steps:
#   1/6  Baseline rewriting chains    (rewriting_pipeline_fictionalqa.py)
#   2/6  Answer F1                    (answer_f1_eval_fictionalqa.py, vLLM QA)
#   3/6  BERTScore (roberta-large)
#   4/6  BLEURT (BLEURT-20) + bleurt_answer
#   5/6  Perplexity (OLMo-3.1-32B, bnb 4-bit via HF transformers)
#   6/6  OpenFActScore (AFG=OLMo-3.1-32B bnb 4-bit, AFV=Gemma-3-4B, topic=first-line)
#
# Why bnb 4-bit: bf16 32B is ~64 GB of weights and does not fit on one 48 GB
# A6000; bnb NF4 brings it to ~20 GB so it runs on a single GPU. vLLM bnb
# does NOT support tensor parallelism in current versions, so TP is forced
# to 1. Use CUDA_VISIBLE_DEVICES=0 (or =1) if you want to pin a specific GPU.
#
# Usage:
#   tmux new -s olmo32b_fictionalqa_90q
#   bash scripts/olmo32b/launch/launch_fictionalqa_90q_bnb_homer.sh \
#     2>&1 | tee logs/olmo32b_fictionalqa_90q_bnb.log
#
# Plan: 90 fictions × 4 instruction types × 3 wordings = 1080 chains,
# each chain = 3 rewriting iterations.
set -euo pipefail
source "$(dirname "$0")/env_homer.sh"

N_FICTIONS="${N_FICTIONS:-90}"
QUANT="${QUANT:-bitsandbytes}"
TP="${TP:-1}"

OUT_DIR="results/olmo32b/fictionalqa_90q_bnb"
CHAINS="$OUT_DIR/rewriting_chains_fictionalqa.csv"
F1="$OUT_DIR/rewriting_chains_fictionalqa_answer_f1.csv"

mkdir -p "$OUT_DIR" logs

echo ""
echo "### 1/6 — Baseline rewriting chains ($N_FICTIONS fictions, OLMo-3.1-32B bnb 4-bit vLLM)"
python scripts/olmo32b/fictionalqa/rewriting_pipeline_fictionalqa.py \
  --n-fictions "$N_FICTIONS" \
  --n-iterations 3 \
  --temperature 0.7 \
  --max-new-tokens 4096 \
  --quantization "$QUANT" \
  --tensor-parallel-size "$TP" \
  --output "$CHAINS"

echo ""
echo "### 2/6 — Answer F1 (QA = OLMo-3.1-32B bnb 4-bit vLLM)"
python scripts/olmo32b/fictionalqa/answer_f1_eval_fictionalqa.py \
  --input "$CHAINS" --output "$F1" \
  --quantization "$QUANT" \
  --tensor-parallel-size "$TP"

echo ""
echo "### 3/6 — BERTScore (roberta-large)"
python scripts/olmo32b/_common/bertscore_eval.py \
  --input "$CHAINS" --batch-size 32

echo ""
echo "### 4/6 — BLEURT (BLEURT-20) + bleurt_answer"
python scripts/olmo32b/_common/bleurt_eval.py \
  --input "$CHAINS" --f1-csv "$F1" --batch-size 64

echo ""
echo "### 5/6 — Perplexity (OLMo-3.1-32B, bnb 4-bit via HF transformers)"
python scripts/olmo32b/_common/perplexity_eval.py --input "$CHAINS" --use-4bit

echo ""
echo "### 6/6 — OpenFActScore (AFG=OLMo-3.1-32B bnb 4-bit, AFV=Gemma-3-4B, topic=first-line)"
python scripts/olmo32b/_common/openfactscore_eval.py \
  --input "$CHAINS" --topic-mode first-line --use-4bit

echo ""
echo "DONE. Baseline RQ1 outputs in $OUT_DIR/"
