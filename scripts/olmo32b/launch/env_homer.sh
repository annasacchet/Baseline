#!/bin/bash
# Shared environment for all OLMo-3.1-32B-Instruct jobs on Homer.
# Source this from the per-dataset launchers: `source env_homer.sh`.

# HF cache lives on NAS (large + persistent). OLMo-3.1-32B is ~64 GB on disk.
export HF_HOME="/mnt/dmif-nas/mitel/sacchet/hf_cache"
export HF_HUB_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache/hub"
export TRANSFORMERS_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache"

# OLMo-3.1-32B-Instruct is open-weights — HF_TOKEN is not strictly required.
# (Set one anyway if you want to avoid HF rate limits.)

# Dataset paths on NAS.
export MUSIQUE_DATASET="${MUSIQUE_DATASET:-/home/sacchet/Baseline/musique_ans_v1.0_dev.jsonl}"
export NEWSQA_DATASET="${NEWSQA_DATASET:-/mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv}"

# No conda activation here — use whichever python is already on $PATH.
# (If you're inside `(base)`, that's miniconda's base env; that's fine.)
echo "Python: $(command -v python) ($(python --version 2>&1))"

cd ~/Baseline

echo "=== OLMo-3.1-32B-Instruct env ==="
echo "HF_HOME         : $HF_HOME"
echo "MUSIQUE_DATASET : $MUSIQUE_DATASET"
echo "NEWSQA_DATASET  : $NEWSQA_DATASET"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null | sed 's/^/  GPU: /'
echo "================================="
