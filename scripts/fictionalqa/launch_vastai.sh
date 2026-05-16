#!/bin/bash
# Launch FictionalQA rewriting pipeline (100q) on vast.ai 2× RTX 3090
# Instance: 36880580 — 151.237.25.234
#
# First-time setup on the instance:
#   pip install transformers accelerate bitsandbytes datasets huggingface_hub pandas torch
#
# Run with:
#   ssh -p <port> root@151.237.25.234 'bash ~/Baseline/scripts/fictionalqa/launch_vastai.sh'

set -e

# HF cache on local NVMe (101 GB available, fast)
export HF_HOME="/root/.cache/huggingface"
export HF_HUB_CACHE="/root/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="/root/.cache/huggingface"

cd ~/Baseline

echo "=== FictionalQA Pipeline — 100q on vast.ai 2× RTX 3090 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python3 scripts/fictionalqa/rewriting_pipeline_fictionalqa.py \
  --model allenai/OLMo-3.1-32B-Instruct \
  --output results/fictionalqa/rewriting_chains_fictionalqa_100q.csv \
  --n-fictions 100 \
  --n-iterations 3 \
  --temperature 0.7 \
  --max-new-tokens 4096 \
  --seed 42 \
  --use-4bit

echo "✓ Done. Output: results/fictionalqa/rewriting_chains_fictionalqa_100q.csv"
