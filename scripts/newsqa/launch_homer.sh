#!/bin/bash
# Launch NewsQA rewriting pipeline smoke test on Homer (2× RTX A6000 48GB each)
# Run this via: ssh sacchet@homer.dimi.uniud.it 'bash ~/Baseline/scripts/newsqa/launch_homer.sh'

set -e

# HF cache — can use /home or NAS
export HF_HOME="/mnt/dmif-nas/mitel/sacchet/hf_cache"
export HF_HUB_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache/hub"
export TRANSFORMERS_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache"

# NewsQA dataset path
export NEWSQA_DATASET="/mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv"

# Go to repo
cd ~/Baseline

# Activate conda
source ~/.bashrc
conda activate baseline 2>/dev/null || conda activate base

echo "=== NewsQA Pipeline Smoke Test on Homer ==="
echo "HF_HOME: $HF_HOME"
echo "NEWSQA_DATASET: $NEWSQA_DATASET"
echo ""

# Smoke test: 1 story, all instructions, bf16 (no quantization)
python scripts/newsqa/rewriting_pipeline_newsqa.py \
  --model allenai/OLMo-3.1-32B-Instruct \
  --dataset "$NEWSQA_DATASET" \
  --output results/newsqa/rewriting_chains_newsqa_smoke.csv \
  --n-questions 1 \
  --n-iterations 3 \
  --temperature 0.7 \
  --max-new-tokens 4096 \
  --seed 42 \
  --smoke-test

echo "✓ Smoke test complete. Output: results/newsqa/rewriting_chains_newsqa_smoke.csv"
