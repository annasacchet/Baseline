#!/bin/bash
# MuSiQue 600q — eval PESANTI (OpenFActScore + Answer F1) sulle qid del rewriting.
# Caricano un LLM 32B in 4-bit (~18-20 GB VRAM): NON lanciare in parallelo al
# rewriting su una RTX 3090 (24 GB) — andrebbe in OOM.
# Lanciare solo quando la GPU è libera (rewriting finito o in pausa).
# Resumable: rilanciandole riprendono dai qid mancanti.
#
# Usage:
#   tmux new -s musique600_eval_heavy
#   bash scripts/600q/launch_eval_heavy_lisa.sh

set -e

export HF_HOME="/mnt/dmif-nas/mitel/sacchet/hf_cache"
export HF_HUB_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache/hub"
export TRANSFORMERS_CACHE="/mnt/dmif-nas/mitel/sacchet/hf_cache"

CHAINS="results/600q/rewriting_chains_musique_600q.csv"
F1_OUT="results/600q/rewriting_chains_musique_600q_answer_f1.csv"
MUSIQUE_JSONL="musique_ans_v1.0_dev.jsonl"

cd ~/Baseline
mkdir -p results/600q
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate baseline 2>/dev/null || conda activate base 2>/dev/null || true

echo "=========================================="
echo "MuSiQue 600q — eval pesanti (OFS + Answer F1)"
echo "Chains: $CHAINS"
echo "=========================================="

# --- 1. OpenFActScore (usa step 0 come fonte, non la gold answer) ---
echo ""
echo "[1/2] OpenFActScore..."
python3 scripts/newsqa/openfactscore_eval_newsqa.py \
  --input "$CHAINS" \
  --use-4bit

# --- 2. Answer F1 (gold answer da musique_ans_v1.0_dev.jsonl tramite qid) ---
echo ""
if [ -f "$MUSIQUE_JSONL" ]; then
  echo "[2/2] Answer F1 (gold da $MUSIQUE_JSONL)..."
  python3 scripts/15q/answer_f1_eval.py \
    --input "$CHAINS" \
    --output "$F1_OUT" \
    --dataset "$MUSIQUE_JSONL" \
    --use-4bit
else
  echo "[2/2] SKIP Answer F1 — manca $MUSIQUE_JSONL nella root del repo."
fi

echo ""
echo "=========================================="
echo "Eval pesanti 600q completate."
echo "=========================================="