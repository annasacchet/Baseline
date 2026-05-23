#!/bin/bash
# Reclassify NOT_SUPPORTED facts from the 600q OFS run with Gemma-3-4B-it (4-bit).
# Same AFV model used for the OFS precision + recall evaluation.
# Default target: the single qid currently available with OFS+recall+F1 (2hop__14092_8311).
#
# Usage:
#   tmux new -s reclassify_600q
#   bash scripts/600q/launch_factscore_reclassify_lisa.sh           # default qid, full run
#   bash scripts/600q/launch_factscore_reclassify_lisa.sh --smoke   # only first 5 facts
#
# Output: results/600q/rewriting_chains_musique_600q_reclassified.csv

set -e

export HF_HOME="/mnt/dmif-nas/mitel/sacchet/hf_cache"
export HF_HUB_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache/hub"
export TRANSFORMERS_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache"

cd ~/Baseline
mkdir -p results/600q
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate baseline 2>/dev/null || conda activate base 2>/dev/null || true

EXTRA_ARGS=""
if [[ "${1:-}" == "--smoke" ]]; then
  EXTRA_ARGS="--limit 5"
  echo "*** SMOKE TEST: 5 facts ***"
fi

echo "=========================================="
echo "600q — reclassify NOT_SUPPORTED facts"
echo "  Model: google/gemma-3-4b-it (4-bit NF4) — same AFV used for OFS+recall"
echo "  QID:   2hop__14092_8311 (default)"
echo "=========================================="

python3 scripts/600q/factscore_reclassify_600q.py \
  --use-4bit \
  $EXTRA_ARGS

echo ""
echo "=========================================="
echo "Done."
echo "  Output: results/600q/rewriting_chains_musique_600q_reclassified.csv"
echo "=========================================="
