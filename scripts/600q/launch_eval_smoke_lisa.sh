#!/bin/bash
# MuSiQue 600q — eval LEGGERE (BERTScore + BLEURT) sulle qid già prodotte dal rewriting.
# Sicure da lanciare in parallelo al rewriting: ~2-4 GB VRAM ciascuna, stanno su una 3090.
# Resumable: rilanciandole a run completato riprendono dai qid mancanti.
#
# Per OFS e Answer F1 (LLM 32B 4-bit, ~18-20 GB) usare launch_eval_heavy_lisa.sh
# quando la GPU è libera (rewriting finito o in pausa).
#
# Usage:
#   tmux new -s musique600_eval
#   bash scripts/600q/launch_eval_smoke_lisa.sh

set -e

export HF_HOME="/mnt/dmif-nas/mitel/sacchet/hf_cache"
export HF_HUB_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache/hub"
export TRANSFORMERS_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache"

CHAINS="results/600q/rewriting_chains_musique_600q.csv"
BERT_OUT="results/600q/rewriting_chains_musique_600q_bertscore.csv"
BLEURT_OUT="results/600q/rewriting_chains_musique_600q_bleurt.csv"

cd ~/Baseline
mkdir -p results/600q
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate baseline 2>/dev/null || conda activate base 2>/dev/null || true

echo "=========================================="
echo "MuSiQue 600q — eval leggere (BERTScore + BLEURT)"
echo "Chains: $CHAINS"
echo "=========================================="

# --- 1. BERTScore (roberta-large, ~2-3 GB) ---
echo ""
echo "[1/2] BERTScore..."
python3 scripts/newsqa/bertscore_eval_newsqa.py \
  --input "$CHAINS" \
  --output "$BERT_OUT" \
  --batch-size 32

# --- 2. BLEURT (baseline + consecutive; --no-answer perché non abbiamo ancora F1) ---
echo ""
echo "[2/2] BLEURT..."
python3 scripts/newsqa/bleurt_eval_newsqa.py \
  --input "$CHAINS" \
  --output "$BLEURT_OUT" \
  --batch-size 64 \
  --no-answer

echo ""
echo "=========================================="
echo "Eval leggere 600q completate."
echo "  BERTScore: $BERT_OUT"
echo "  BLEURT:    $BLEURT_OUT"
echo "=========================================="