#!/bin/bash
# FINAL experiment — Qwen3-30B-A3B-Instruct-2507 · FictionalQA · 600 questions.
# Server: LISA. Whole pipeline (forward + OFS + recall) locally.
# FictionalQA = one question per fiction document, pulled from the HF Hub.
#
# Fully RESUMABLE. Usage:
#   tmux new -s qwen_fictionalqa
#   bash scripts/final_experiments/qwen30b_lisa/run_qwen_fictionalqa_600q.sh \
#       2>&1 | tee logs/final_qwen_fictionalqa_600q.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_lisa.sh"
source "$HERE/../_env/lib_forward.sh"
source "$HERE/../_env/lib_ofs.sh"

export MODEL_DIR="qwen30b"
export DATASET="fictionalqa"
export TAG="fictionalqa_600q"
export OUT_DIR="${OUT_DIR:-results/final/qwen30b/fictionalqa_600q}"

export N_ITEMS="${N_ITEMS:-600}"
export N_ITERATIONS="${N_ITERATIONS:-3}"
export BACKEND="${BACKEND:-vllm}"
export QUANT="${QUANT:-}"
export TP="${TP:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

run_forward

export CHAINS="$OUT_DIR/rewriting_chains_${TAG}.csv"
export TOPIC_MODE=first-line
run_ofs_recall

echo ""
echo "ALL DONE — Qwen FictionalQA 600q: $OUT_DIR"
