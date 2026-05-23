#!/bin/bash
# Reclassify NOT_SUPPORTED facts from the 300q OFS run with Gemma-3-4B-it (4-bit).
# Same AFV model used for the OFS labels, 5-category prompt (with SUPPORTED
# escape for AFV false positives).
#
# Targets the 10 qid (4-hop) for which per-claim OFS details are already
# available in the 300q output. 2-hop and 3-hop qids don't have details
# saved on disk so they are skipped (would require re-running OFS forward).
#
# Usage:
#   tmux new -s reclassify_300q
#   bash scripts/300q/launch_factscore_reclassify_lisa.sh           # full run
#   bash scripts/300q/launch_factscore_reclassify_lisa.sh --smoke   # 5 facts
#
# Output: results/300q/rewriting_chains_300q_reclassified.csv  (resumable)

set -e

export HF_HOME="/mnt/dmif-nas/mitel/sacchet/hf_cache"
export HF_HUB_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache/hub"
export TRANSFORMERS_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache"

cd ~/Baseline
mkdir -p results/300q
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate baseline 2>/dev/null || conda activate base 2>/dev/null || true

EXTRA_ARGS=""
if [[ "${1:-}" == "--smoke" ]]; then
  EXTRA_ARGS="--limit 5"
  echo "*** SMOKE TEST: 5 facts ***"
fi

# 10 4-hop qids available in the existing details file
QIDS=(
  "4hop3__630723_88460_30152_20999"
  "4hop3__650050_88460_30152_20999"
  "4hop3__661103_698586_1926_54362"
  "4hop3__719125_132409_223216_35031"
  "4hop3__678814_466199_695123_72134"
  "4hop3__638064_88460_30152_20999"
  "4hop3__771862_466199_695123_72134"
  "4hop3__862_846_326964_7713"
  "4hop3__794915_466199_695123_72134"
  "4hop3__857_846_613770_7713"
)

QID_ARGS=""
for q in "${QIDS[@]}"; do
  QID_ARGS="$QID_ARGS --qid $q"
done

echo "=========================================="
echo "300q — reclassify NOT_SUPPORTED facts"
echo "  Model: google/gemma-3-4b-it (4-bit NF4) — same AFV as OFS"
echo "  Categories: 5 (SUPPORTED, CONTRADICTION, DISTORTED, INVENTED, UNVERIFIABLE)"
echo "  QIDs: 10 4-hop"
echo "=========================================="

python3 scripts/300q/factscore_reclassify_300q.py \
  --use-4bit \
  $QID_ARGS \
  $EXTRA_ARGS

echo ""
echo "=========================================="
echo "Done."
echo "  Output: results/300q/rewriting_chains_300q_reclassified.csv"
echo "=========================================="
