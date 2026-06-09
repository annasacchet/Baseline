#!/bin/bash
# FINAL experiment — Qwen3-30B-A3B-Instruct-2507 · NewsQA · 600 questions.
# Server: LISA. Whole pipeline (forward + OFS + recall) locally.
# NewsQA recall is NLI-based; OFS forward uses Gemma AFV.
#
# Fully RESUMABLE. Usage:
#   tmux new -s qwen_newsqa
#   bash scripts/final_experiments/qwen30b_lisa/run_qwen_newsqa_600q.sh \
#       2>&1 | tee logs/final_qwen_newsqa_600q.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_lisa.sh"
source "$HERE/../_env/lib_forward.sh"
source "$HERE/../_env/lib_ofs.sh"

export MODEL_DIR="qwen30b"
export DATASET="newsqa"
export TAG="newsqa_600q"
export OUT_DIR="${OUT_DIR:-results/final/qwen30b/newsqa_600q}"

export N_ITEMS="${N_ITEMS:-600}"
export N_ITERATIONS="${N_ITERATIONS:-3}"
export BACKEND="${BACKEND:-vllm}"
export QUANT="${QUANT:-}"
export TP="${TP:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"

run_forward

export CHAINS="$OUT_DIR/rewriting_chains_${TAG}.csv"
export TOPIC_MODE=first-line
run_ofs_recall

echo ""
echo "ALL DONE — Qwen NewsQA 600q: $OUT_DIR"
