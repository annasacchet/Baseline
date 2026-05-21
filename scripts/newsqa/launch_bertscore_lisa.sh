#!/bin/bash
# NewsQA 100q BERTScore — Lisa
# Usage:
#   tmux new -s newsqa_bs
#   bash scripts/newsqa/launch_bertscore_lisa.sh

set -e

export HF_HOME="/mnt/dmif-nas/mitel/sacchet/hf_cache"
export HF_HUB_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache/hub"
export TRANSFORMERS_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache"

CHAINS="results/newsqa/rewriting_chains_newsqa_100q.csv"
OUT="results/newsqa/rewriting_chains_newsqa_100q_bertscore.csv"
PILOT_BACKUP="results/newsqa/rewriting_chains_newsqa_100q_bertscore_pilot7qid.csv"

cd ~/Baseline
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate baseline 2>/dev/null || conda activate base 2>/dev/null || true

echo "=========================================="
echo "NewsQA 100q BERTScore — Lisa"
echo "  model: roberta-large (layer 17)"
echo "  chains: $CHAINS"
echo "  output: $OUT"
echo "=========================================="

# The pre-existing CSV only has 7 qids (pilot). Move it aside before the full run
# so we start from a clean slate (the script's resume logic would otherwise keep them).
if [ -f "$OUT" ]; then
  rows=$(wc -l < "$OUT")
  if [ "$rows" -lt 1000 ]; then
    echo "Found small/pilot BERTScore file ($rows lines) — moving to $PILOT_BACKUP"
    mv "$OUT" "$PILOT_BACKUP"
  else
    echo "Found existing BERTScore file with $rows lines — will resume."
  fi
fi

python3 scripts/newsqa/bertscore_eval_newsqa.py \
  --input "$CHAINS" \
  --output "$OUT" \
  --batch-size 32

echo ""
echo "=========================================="
echo "BERTScore complete: $OUT"
echo "=========================================="
