#!/bin/bash
# FINAL experiment — Qwen3-30B-A3B-Instruct-2507 · MuSiQue · 600 questions (200/hop).
# Server: LISA. Whole pipeline (forward + OFS + recall) locally.
#
# Qwen3-30B-A3B is MoE (~30B total / ~3B active) and HAS a native vLLM kernel,
# so rewriting/F1 run on vLLM (no HF fallback needed). It is the non-thinking
# Instruct checkpoint -> no thinking-block stripping.
#
# Metrics: rewriting · BERTScore (baseline+consecutive) · BLEURT · Answer F1 ·
#          FactScore (OFS) · Recall · token length.
#
# Fully RESUMABLE. Usage:
#   tmux new -s qwen_musique
#   bash scripts/final_experiments/qwen30b_lisa/run_qwen_musique_600q.sh \
#       2>&1 | tee logs/final_qwen_musique_600q.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_lisa.sh"
source "$HERE/../_env/lib_forward.sh"
source "$HERE/../_env/lib_ofs.sh"

export MODEL_DIR="qwen30b"
export DATASET="musique"
export TAG="musique_600q"
export OUT_DIR="${OUT_DIR:-results/final/qwen30b/musique_600q}"

export N_PER_HOP="${N_PER_HOP:-200}"      # 200/hop -> 600 questions
export N_ITERATIONS="${N_ITERATIONS:-3}"
export BACKEND="${BACKEND:-vllm}"         # Qwen3 has a native vLLM kernel
export QUANT="${QUANT:-bitsandbytes}"   # NF4 4-bit: Qwen3-30B entra in 24GB (bf16 -> OOM)
export TP="${TP:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

run_forward

export CHAINS="$OUT_DIR/rewriting_chains_${TAG}.csv"
export TOPIC_MODE=qid
run_ofs_recall

echo ""
echo "ALL DONE — Qwen MuSiQue 600q: $OUT_DIR"
