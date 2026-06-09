#!/bin/bash
# FINAL experiment — OLMo-3.1-32B-Instruct · FictionalQA · 600 questions.
# Server: LISA. Whole pipeline (forward + OFS + recall) locally.
# FictionalQA is one question per fiction document, pulled from the HF Hub.
#
# Metrics: rewriting · BERTScore (baseline+consecutive) · BLEURT · Answer F1 ·
#          FactScore (OFS, Gemma AFV) · Recall (Gemma AFV) · token length.
#
# Fully RESUMABLE. Usage:
#   tmux new -s olmo_fictionalqa
#   bash scripts/final_experiments/olmo32b_lisa/run_olmo_fictionalqa_600q.sh \
#       2>&1 | tee logs/final_olmo_fictionalqa_600q.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_lisa.sh"
source "$HERE/../_env/lib_forward.sh"
source "$HERE/../_env/lib_ofs.sh"

export MODEL_DIR="olmo32b"
export DATASET="fictionalqa"
export TAG="fictionalqa_600q"
export OUT_DIR="${OUT_DIR:-results/final/olmo32b/fictionalqa_600q}"

export N_ITEMS="${N_ITEMS:-600}"          # 600 fiction docs -> 600 questions
export N_ITERATIONS="${N_ITERATIONS:-3}"
export BACKEND="${BACKEND:-hf}"
export QUANT="${QUANT:-bitsandbytes}"
export TP="${TP:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export USE_4BIT_PPL=1

run_forward

export CHAINS="$OUT_DIR/rewriting_chains_${TAG}.csv"
export TOPIC_MODE=first-line
run_ofs_recall

echo ""
echo "ALL DONE — OLMo FictionalQA 600q: $OUT_DIR"
