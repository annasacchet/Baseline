#!/bin/bash
# Shared environment for all Llama-3.1-70B jobs on Homer.
# Source this from the per-dataset launchers: `source env_homer.sh`.

# HF cache lives on NAS (large + persistent). Llama-3.1-70B is ~140 GB on disk.
export HF_HOME="/mnt/dmif-nas/mitel/sacchet/hf_cache"
export HF_HUB_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache/hub"
export TRANSFORMERS_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache"

# meta-llama/Llama-3.1-70B-Instruct is a gated repo — HF_TOKEN must be set.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[WARN] HF_TOKEN is not set — Llama-3.1-70B is gated and the load will fail." >&2
  echo "       Set it via:  export HF_TOKEN=hf_..." >&2
fi

# Dataset paths on NAS.
export MUSIQUE_DATASET="${MUSIQUE_DATASET:-/home/sacchet/Baseline/musique_ans_v1.0_dev.jsonl}"
export NEWSQA_DATASET="${NEWSQA_DATASET:-/mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv}"

# Conda env (created with bitsandbytes + transformers >= 4.45 for Llama-3.1).
# Source conda.sh directly so `activate` works under `set -e` and even when
# `conda init` was never wired into ~/.bashrc.
_find_conda_sh() {
  for cand in \
      "$HOME/miniconda3/etc/profile.d/conda.sh" \
      "$HOME/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh" \
      "/opt/miniconda3/etc/profile.d/conda.sh" \
      "/usr/local/miniconda3/etc/profile.d/conda.sh" \
      "/usr/local/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$cand" ]; then echo "$cand"; return 0; fi
  done
  if command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base 2>/dev/null || true)"
    if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
      echo "$base/etc/profile.d/conda.sh"; return 0
    fi
  fi
  return 1
}

CONDA_SH="$(_find_conda_sh || true)"
if [ -n "$CONDA_SH" ]; then
  echo "Loading conda from $CONDA_SH"
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate baseline 2>/dev/null \
    || conda activate base 2>/dev/null \
    || echo "[WARN] could not activate 'baseline' or 'base' — using current PATH" >&2
else
  echo "[WARN] conda.sh not found in any known location — using current PATH" >&2
fi
echo "Python: $(command -v python) ($(python --version 2>&1))"

cd ~/Baseline

echo "=== Llama-3.1-70B env ==="
echo "HF_HOME         : $HF_HOME"
echo "MUSIQUE_DATASET : $MUSIQUE_DATASET"
echo "NEWSQA_DATASET  : $NEWSQA_DATASET"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null | sed 's/^/  GPU: /'
echo "========================"
