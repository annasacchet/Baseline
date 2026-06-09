#!/bin/bash
# OFS + RECALL for gpt-oss-120b · NewsQA 600q — runs on LISA.
# Forward OFS uses Gemma AFV; NewsQA recall is NLI-based. Operates on Homer chains.
#
# Fully RESUMABLE. Usage:
#   tmux new -s gptoss_newsqa_ofs
#   bash scripts/final_experiments/gptoss120b_homer/run_gptoss_newsqa_600q_ofs_recall.sh \
#       2>&1 | tee logs/final_gptoss_newsqa_600q_ofs_recall.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_lisa.sh"
source "$HERE/../_env/lib_ofs.sh"

export MODEL_DIR="gptoss120b"
export DATASET="newsqa"
export OUT_DIR="${OUT_DIR:-results/final/gptoss120b/newsqa_600q}"
export CHAINS="${CHAINS:-$OUT_DIR/rewriting_chains_newsqa_600q.csv}"
export TOPIC_MODE=first-line
export AFV_USE_4BIT=1

run_ofs_recall

echo ""
echo "ALL DONE — gpt-oss NewsQA 600q OFS+recall: $OUT_DIR"
