#!/bin/bash
# FINAL experiment — OLMo-3.1-32B-Instruct · NewsQA · 600 questions.
# Server: LISA. Whole pipeline (forward + OFS + recall) locally.
#
# Metrics: rewriting · BERTScore (baseline+consecutive) · BLEURT · Answer F1 ·
#          FactScore (OFS) · Recall (NLI cross-encoder) · token length.
# NewsQA recall is NLI-based (not Gemma AFV); OFS forward still uses Gemma.
#
# Fully RESUMABLE. Usage:
#   tmux new -s olmo_newsqa
#   bash scripts/final_experiments/olmo32b_lisa/run_olmo_newsqa_600q.sh \
#       2>&1 | tee logs/final_olmo_newsqa_600q.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_lisa.sh"
source "$HERE/../_env/lib_forward.sh"
source "$HERE/../_env/lib_ofs.sh"

export MODEL_DIR="olmo32b"
export DATASET="newsqa"
export TAG="newsqa_600q"
export OUT_DIR="${OUT_DIR:-results/final/olmo32b/newsqa_600q}"

export N_ITEMS="${N_ITEMS:-600}"
export N_ITERATIONS="${N_ITERATIONS:-3}"
export BACKEND="${BACKEND:-hf}"
export QUANT="${QUANT:-bitsandbytes}"
export TP="${TP:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
export USE_4BIT_PPL=1

run_forward

export CHAINS="$OUT_DIR/rewriting_chains_${TAG}.csv"
export TOPIC_MODE=first-line   # NewsQA E_0 has no qid topic string
run_ofs_recall

echo ""
echo "ALL DONE — OLMo NewsQA 600q: $OUT_DIR"
