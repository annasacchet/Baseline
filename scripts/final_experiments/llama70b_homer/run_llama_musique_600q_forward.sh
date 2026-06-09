#!/bin/bash
# FINAL experiment — Llama-3.1-70B-Instruct · MuSiQue · 600q (200/hop) · FORWARD only.
# Server: HOMER. Runs everything EXCEPT factscore + recall.
#
# Llama-70B is served as the AWQ-INT4 checkpoint via vLLM (the rewriting and
# musique Answer-F1 pipelines are vLLM-only — no --backend/--quantization flags).
#
# Metrics here: rewriting · Answer F1 · BERTScore (baseline+consecutive) ·
#               BLEURT · Perplexity · token length.
# OFS + recall are a SEPARATE step on Lisa:
#   scripts/final_experiments/llama70b_homer/run_llama_musique_600q_ofs_recall.sh
#
# Fully RESUMABLE. Usage:
#   tmux new -s llama_musique
#   bash scripts/final_experiments/llama70b_homer/run_llama_musique_600q_forward.sh \
#       2>&1 | tee logs/final_llama_musique_600q_forward.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_homer.sh"
source "$HERE/../_env/lib_forward.sh"

# Activate the vLLM env that serves the Llama AWQ checkpoint, if present.
if command -v conda >/dev/null 2>&1; then
  CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  [ -f "$CONDA_SH" ] && source "$CONDA_SH"
  conda activate vllm311 2>/dev/null || echo "[WARN] couldn't activate vllm311" >&2
fi

export MODEL_DIR="llama70b"
export DATASET="musique"
export TAG="musique_600q"
export OUT_DIR="${OUT_DIR:-results/final/llama70b/musique_600q}"

export N_PER_HOP="${N_PER_HOP:-200}"      # 200/hop -> 600 questions
export N_ITERATIONS="${N_ITERATIONS:-3}"
export BACKEND=""                          # llama pipelines are vLLM/AWQ-only
export QUANT=""
export TP="${TP:-2}"                        # 2x A6000 on Homer
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3500}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export USE_4BIT_PPL=1                       # 70B perplexity via HF bnb 4-bit

run_forward

echo ""
echo "FORWARD DONE — Llama MuSiQue 600q: $OUT_DIR"
echo "Next: run OFS + recall on LISA against $OUT_DIR/rewriting_chains_${TAG}.csv"
echo "  bash scripts/final_experiments/llama70b_homer/run_llama_musique_600q_ofs_recall.sh"
