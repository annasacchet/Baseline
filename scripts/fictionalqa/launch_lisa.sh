#!/bin/bash
# Launch FictionalQA rewriting pipeline smoke test on Lisa (2× RTX 3090)
# Run this via: ssh sacchet@lisa.dimi.uniud.it 'bash ~/Baseline/scripts/fictionalqa/launch_lisa.sh'

set -e

# HF cache — must live on NAS on Lisa, not /home
export HF_HOME="/mnt/dmif-nas/mitel/sacchet/hf_cache"
export HF_HUB_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache/hub"
export TRANSFORMERS_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache"

# Go to repo
cd ~/Baseline

# Activate conda (adjust env name if different)
source ~/.bashrc
conda activate baseline 2>/dev/null || conda activate base

echo "=== FictionalQA Pipeline Smoke Test on Lisa ==="
echo "HF_HOME: $HF_HOME"
echo ""

# Smoke test: 1 fiction, all instructions
python scripts/fictionalqa/rewriting_pipeline_fictionalqa.py \
  --model allenai/OLMo-3.1-32B-Instruct \
  --output results/fictionalqa/rewriting_chains_fictionalqa_smoke.csv \
  --n-fictions 1 \
  --n-iterations 3 \
  --temperature 0.7 \
  --max-new-tokens 4096 \
  --seed 42 \
  --smoke-test \
  --use-4bit

echo "✓ Smoke test complete. Output: results/fictionalqa/rewriting_chains_fictionalqa_smoke.csv"
