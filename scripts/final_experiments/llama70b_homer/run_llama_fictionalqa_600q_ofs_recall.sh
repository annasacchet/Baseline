#!/bin/bash
# OFS + RECALL for Llama-3.1-70B · FictionalQA 600q — runs on LISA.
# Forward OFS + recall both use Gemma AFV. Operates on the Homer chains.
#
# Fully RESUMABLE. Usage:
#   tmux new -s llama_fictionalqa_ofs
#   bash scripts/final_experiments/llama70b_homer/run_llama_fictionalqa_600q_ofs_recall.sh \
#       2>&1 | tee logs/final_llama_fictionalqa_600q_ofs_recall.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_lisa.sh"
source "$HERE/../_env/lib_ofs.sh"

export MODEL_DIR="llama70b"
export DATASET="fictionalqa"
export OUT_DIR="${OUT_DIR:-results/final/llama70b/fictionalqa_600q}"
export CHAINS="${CHAINS:-$OUT_DIR/rewriting_chains_fictionalqa_600q.csv}"
export TOPIC_MODE=first-line
export AFV_USE_4BIT=1

run_ofs_recall

echo ""
echo "ALL DONE — Llama FictionalQA 600q OFS+recall: $OUT_DIR"
