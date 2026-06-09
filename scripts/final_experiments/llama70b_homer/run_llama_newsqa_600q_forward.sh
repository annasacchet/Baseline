#!/bin/bash
# FINAL experiment — Llama-3.1-70B-Instruct · NewsQA · 600q · FORWARD only.
# Server: HOMER. Everything EXCEPT factscore + recall.
#
# NOTE: the Llama NewsQA Answer-F1 evaluator is HF/bnb 4-bit (not vLLM), so
# It takes --batch-size/--use-4bit, no --max-model-len/--resume; the launcher detects this automatically.
#
# OFS + recall run separately on Lisa:
#   scripts/final_experiments/llama70b_homer/run_llama_newsqa_600q_ofs_recall.sh
#
# Fully RESUMABLE. Usage:
#   tmux new -s llama_newsqa
#   bash scripts/final_experiments/llama70b_homer/run_llama_newsqa_600q_forward.sh \
#       2>&1 | tee logs/final_llama_newsqa_600q_forward.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_homer.sh"
source "$HERE/../_env/lib_forward.sh"

if command -v conda >/dev/null 2>&1; then
  CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  [ -f "$CONDA_SH" ] && source "$CONDA_SH"
  conda activate vllm311 2>/dev/null || echo "[WARN] couldn't activate vllm311" >&2
fi

export MODEL_DIR="llama70b"
export DATASET="newsqa"
export TAG="newsqa_600q"
export OUT_DIR="${OUT_DIR:-results/final/llama70b/newsqa_600q}"

export N_ITEMS="${N_ITEMS:-600}"
export N_ITERATIONS="${N_ITERATIONS:-3}"
export BACKEND=""
export QUANT=""
export TP="${TP:-2}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
export USE_4BIT_PPL=1

run_forward

echo ""
echo "FORWARD DONE — Llama NewsQA 600q: $OUT_DIR"
echo "Next: run OFS + recall on LISA:"
echo "  bash scripts/final_experiments/llama70b_homer/run_llama_newsqa_600q_ofs_recall.sh"
