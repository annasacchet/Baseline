#!/bin/bash
# Shared environment for the FINAL experiments running on Homer.
#
# Homer hosts the two large models for the FORWARD pipeline only:
#   * Llama-3.1-70B-Instruct  (served via vLLM as the AWQ-INT4 checkpoint)
#   * openai/gpt-oss-120b     (served via vLLM, native)
#
# On Homer we run everything EXCEPT factscore + recall:
#   rewriting -> Answer F1 -> BERTScore -> BLEURT -> Perplexity (token length is
#   captured inside the rewriting step as n_tokens).
# OpenFActScore + recall for these two models are run SEPARATELY on Lisa, on the
# chains produced here (the AFV judge Gemma-3-4B is small and lives on Lisa).
set -euo pipefail

# HF cache su NAS. Rispettiamo eventuali export già fatti nella shell
# (${VAR:-default}), così se li imposti tu prima del comando vincono i tuoi.
export HF_HOME="${HF_HOME:-/mnt/dmif-nas/mitel/sacchet/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/dmif-nas/mitel/sacchet/hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/dmif-nas/mitel/sacchet/hf_cache/hub}"
export PYTHONUNBUFFERED=1
# Riduce la frammentazione della VRAM (suggerito da PyTorch sugli OOM).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# NIENTE conda: attiva il tuo venv PRIMA di lanciare (es. quello che serve vLLM
# per Llama-AWQ / gpt-oss). Se nel PATH c'è solo `python3`, shim `python`→`python3`.
if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  python() { python3 "$@"; }
  export -f python 2>/dev/null || true
  echo "[env] 'python' non trovato → uso 'python3' ($(command -v python3))"
fi

# meta-llama/Llama-3.1-70B-Instruct is gated → HF_TOKEN required.
# openai/gpt-oss-120b is open-weights; a token only avoids HF rate limits.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[WARN] HF_TOKEN not set — Llama-3.1-70B is gated and its load will fail." >&2
  echo "       export HF_TOKEN=hf_..." >&2
fi

export MUSIQUE_DATASET="${MUSIQUE_DATASET:-/home/sacchet/Baseline/musique_ans_v1.0_dev.jsonl}"
# NewsQA: prova prima ~/datasets (path su Lisa), poi il NAS. Override con NEWSQA_DATASET=...
if [ -z "${NEWSQA_DATASET:-}" ]; then
  if [ -f /home/sacchet/datasets/combined-newsqa-data-v1.csv ]; then
    export NEWSQA_DATASET=/home/sacchet/datasets/combined-newsqa-data-v1.csv
  else
    export NEWSQA_DATASET=/mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv
  fi
fi

cd ~/Baseline

echo "=== FINAL experiments — Homer env ==="
echo "HF_HOME         : $HF_HOME"
echo "MUSIQUE_DATASET : $MUSIQUE_DATASET"
echo "NEWSQA_DATASET  : $NEWSQA_DATASET"
echo "Python          : $(command -v python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null | sed 's/^/  GPU: /'
echo "==================================="
