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

# No conda activation here — use whichever python is already on $PATH.
# (If you're inside `(base)`, that's miniconda's base env; that's fine.)
echo "Python: $(command -v python) ($(python --version 2>&1))"

cd ~/Baseline

echo "=== Llama-3.1-70B env ==="
echo "HF_HOME         : $HF_HOME"
echo "MUSIQUE_DATASET : $MUSIQUE_DATASET"
echo "NEWSQA_DATASET  : $NEWSQA_DATASET"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null | sed 's/^/  GPU: /'
echo "========================"
