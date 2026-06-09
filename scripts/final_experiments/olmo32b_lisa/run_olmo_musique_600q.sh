#!/bin/bash
# FINAL experiment — OLMo-3.1-32B-Instruct · MuSiQue · 600 questions (200/hop).
# Server: LISA. Runs the WHOLE pipeline (forward + OFS + recall) locally.
#
# Metrics: rewriting (iterative, 3 steps) · BERTScore (baseline+consecutive) ·
#          BLEURT · Answer F1 · FactScore (OFS) · Recall · token length (n_tokens).
#
# OLMo-3.1-32B has NO native vLLM kernel (memory: olmo_vllm_no_native_kernel),
# so the rewriting/F1 backend is HF transformers (--backend hf), 4-bit for PPL.
#
# Fully RESUMABLE: re-run this exact command after any interruption and every
# step continues from where it stopped.
#
# Usage:
#   tmux new -s olmo_musique
#   bash scripts/final_experiments/olmo32b_lisa/run_olmo_musique_600q.sh \
#       2>&1 | tee logs/final_olmo_musique_600q.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_lisa.sh"
source "$HERE/../_env/lib_forward.sh"
source "$HERE/../_env/lib_ofs.sh"

export MODEL_DIR="olmo32b"
export DATASET="musique"
export TAG="musique_600q"
export OUT_DIR="${OUT_DIR:-results/final/olmo32b/musique_600q}"

# 200 per hop class (2/3/4-hop) -> 600 questions, balanced.
export N_PER_HOP="${N_PER_HOP:-200}"
export N_ITERATIONS="${N_ITERATIONS:-3}"
export BACKEND="${BACKEND:-hf}"          # OLMo-3.1 -> HF backend (no vLLM kernel)
export QUANT="${QUANT:-bitsandbytes}"    # NF4 4-bit so 32B fits
export TP="${TP:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export USE_4BIT_PPL=1

run_forward

export CHAINS="$OUT_DIR/rewriting_chains_${TAG}.csv"
export TOPIC_MODE=qid
run_ofs_recall

echo ""
echo "ALL DONE — OLMo MuSiQue 600q: $OUT_DIR"
