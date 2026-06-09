#!/bin/bash
# Shared environment for the FINAL experiments running on Lisa.
#
# Lisa hosts:
#   * OLMo-3.1-32B-Instruct   (full forward pipeline + OFS + recall, local)
#   * Qwen3-30B-A3B-Instruct  (full forward pipeline + OFS + recall, local)
#   * the OFS + recall step for the Homer models (Llama-70B, gpt-oss-120b):
#     their forward chains are produced on Homer, then OFS/recall is run here
#     because the AFV judge (Gemma-3-4B) is small and lives on this box.
#
# CRITICAL (see memory feedback_lisa_hf_cache): the HF cache MUST live on the
# NAS, never in /home, or Lisa fills its tiny system disk.
set -euo pipefail

# HF cache su NAS. Rispettiamo eventuali export già fatti nella shell
# (${VAR:-default}), così se li imposti tu prima del comando vincono i tuoi.
export HF_HOME="${HF_HOME:-/mnt/dmif-nas/mitel/sacchet/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/dmif-nas/mitel/sacchet/hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/dmif-nas/mitel/sacchet/hf_cache/hub}"
export PYTHONUNBUFFERED=1

# NIENTE conda: usiamo il python dell'ambiente che hai già attivo (venv o
# sistema). Attiva il tuo venv PRIMA di lanciare, es.:
#     source ~/Baseline/.venv/bin/activate
# I pipeline chiamano `python`. Se nel PATH c'è solo `python3`, creiamo uno shim
# `python` → `python3` così gli script girano comunque.
if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  python() { python3 "$@"; }
  export -f python 2>/dev/null || true
  echo "[env] 'python' non trovato → uso 'python3' ($(command -v python3))"
fi

# Gemma-3-4B-it (AFV judge for OFS + recall) is gated → HF_TOKEN required.
# Qwen3-30B / OLMo-3.1 are open-weights; a token only avoids rate limits.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[WARN] HF_TOKEN not set — Gemma-3-4B (AFV judge) is gated; OFS/recall load may fail." >&2
  echo "       export HF_TOKEN=hf_..." >&2
fi

# Dataset paths. MuSiQue + NewsQA are local files on the NAS / home; FictionalQA
# is pulled from the HF Hub by the pipeline itself (no path needed).
export MUSIQUE_DATASET="${MUSIQUE_DATASET:-/home/sacchet/Baseline/musique_ans_v1.0_dev.jsonl}"
export NEWSQA_DATASET="${NEWSQA_DATASET:-/mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv}"

cd ~/Baseline

echo "=== FINAL experiments — Lisa env ==="
echo "HF_HOME         : $HF_HOME"
echo "MUSIQUE_DATASET : $MUSIQUE_DATASET"
echo "NEWSQA_DATASET  : $NEWSQA_DATASET"
echo "Python          : $(command -v python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null | sed 's/^/  GPU: /'
echo "==================================="
