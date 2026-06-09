#!/bin/bash
# FINAL experiment — openai/gpt-oss-120b · MuSiQue · 600q (200/hop) · FORWARD only.
# Server: HOMER. Runs everything EXCEPT factscore + recall.
#
# gpt-oss-120b is a ~120B MoE (~5B active), served via vLLM with tensor
# parallelism across the Homer GPUs. The cloned pipeline's Answer-F1 is
# vLLM-style for every dataset.
#
# Metrics here: rewriting · Answer F1 · BERTScore (baseline+consecutive) ·
#               BLEURT · Perplexity · token length.
# OFS + recall run separately on Lisa:
#   scripts/final_experiments/gptoss120b_homer/run_gptoss_musique_600q_ofs_recall.sh
#
# Fully RESUMABLE. Usage:
#   tmux new -s gptoss_musique
#   bash scripts/final_experiments/gptoss120b_homer/run_gptoss_musique_600q_forward.sh \
#       2>&1 | tee logs/final_gptoss_musique_600q_forward.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_homer.sh"
source "$HERE/../_env/lib_forward.sh"

if command -v conda >/dev/null 2>&1; then
  CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  [ -f "$CONDA_SH" ] && source "$CONDA_SH"
  conda activate vllm311 2>/dev/null || echo "[WARN] couldn't activate vllm311" >&2
fi

export MODEL_DIR="gptoss120b"
export DATASET="musique"
export TAG="musique_600q"
export OUT_DIR="${OUT_DIR:-results/final/gptoss120b/musique_600q}"

export N_PER_HOP="${N_PER_HOP:-200}"      # 200/hop -> 600 questions
export N_ITERATIONS="${N_ITERATIONS:-3}"
export BACKEND="${BACKEND:-vllm}"
export QUANT="${QUANT:-}"
export TP="${TP:-2}"                        # spread the 120B MoE over both GPUs
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export USE_4BIT_PPL=1

run_forward

echo ""
echo "FORWARD DONE — gpt-oss MuSiQue 600q: $OUT_DIR"
echo "Next: run OFS + recall on LISA:"
echo "  bash scripts/final_experiments/gptoss120b_homer/run_gptoss_musique_600q_ofs_recall.sh"
