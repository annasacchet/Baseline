#!/bin/bash
# 600q — Answer F1 sul singolo qid 2hop__14092_8311 (qid attualmente in analisi).
# Su Homer (2× 48GB). Loader 4-bit, ~18-20GB su una sola GPU.
#
# Output:
#   results/600q/rewriting_chains_musique_600q_answer_f1_2hop__14092_8311.csv
#
# Usage:
#   tmux new -s f1_qid
#   bash scripts/600q/launch_answer_f1_qid_homer.sh

set -e

export HF_HOME="/mnt/dmif-nas/mitel/sacchet/hf_cache"
export HF_HUB_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache/hub"
export TRANSFORMERS_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache"

QID="2hop__14092_8311"
CHAINS="results/600q/rewriting_chains_musique_600q.csv"
MUSIQUE_JSONL="musique_ans_v1.0_dev.jsonl"

cd ~/Baseline
mkdir -p results/600q
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate baseline 2>/dev/null || conda activate base 2>/dev/null || true

echo "=========================================="
echo "MuSiQue 600q — Answer F1 sul qid $QID"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""

if [ ! -f "$MUSIQUE_JSONL" ]; then
  echo "ERROR: manca $MUSIQUE_JSONL nella root del repo (gitignored)."
  echo "Scarica/copia il dev MuSiQue jsonl prima di rilanciare."
  exit 1
fi

python3 scripts/600q/answer_f1_eval_600q.py \
  --input "$CHAINS" \
  --dataset "$MUSIQUE_JSONL" \
  --qid "$QID" \
  --batch-size 8 \
  --use-4bit \
  --resume

echo ""
echo "=========================================="
echo "Done. Output:"
echo "  results/600q/rewriting_chains_musique_600q_answer_f1_${QID}.csv"
echo "=========================================="
