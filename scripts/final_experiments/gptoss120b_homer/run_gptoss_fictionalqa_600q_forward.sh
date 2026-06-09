#!/bin/bash
# FINAL experiment — openai/gpt-oss-120b · FictionalQA · 600q · FORWARD only.
# Server: HOMER. Everything EXCEPT factscore + recall. vLLM backend.
#
# OFS + recall run separately on Lisa:
#   scripts/final_experiments/gptoss120b_homer/run_gptoss_fictionalqa_600q_ofs_recall.sh
#
# Fully RESUMABLE. Usage:
#   tmux new -s gptoss_fictionalqa
#   bash scripts/final_experiments/gptoss120b_homer/run_gptoss_fictionalqa_600q_forward.sh \
#       2>&1 | tee logs/final_gptoss_fictionalqa_600q_forward.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_homer.sh"
source "$HERE/../_env/lib_forward.sh"


export MODEL_DIR="gptoss120b"
export DATASET="fictionalqa"
export TAG="fictionalqa_600q"
export OUT_DIR="${OUT_DIR:-results/final/gptoss120b/fictionalqa_600q}"

export N_ITEMS="${N_ITEMS:-600}"
export N_ITERATIONS="${N_ITERATIONS:-3}"
export BACKEND="${BACKEND:-vllm}"
export QUANT="${QUANT:-}"
export TP="${TP:-2}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export USE_4BIT_PPL=1

run_forward

echo ""
echo "FORWARD DONE — gpt-oss FictionalQA 600q: $OUT_DIR"
echo "Next: run OFS + recall on LISA:"
echo "  bash scripts/final_experiments/gptoss120b_homer/run_gptoss_fictionalqa_600q_ofs_recall.sh"
