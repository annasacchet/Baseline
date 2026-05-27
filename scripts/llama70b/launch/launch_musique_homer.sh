#!/bin/bash
# Full MuSiQue pipeline (rewriting + 5 evals) on Homer with Llama-3.1-70B 4-bit.
# Usage:
#   tmux new -s llama_musique
#   bash scripts/llama70b/launch/launch_musique_homer.sh 2>&1 | tee logs/llama_musique.log
set -euo pipefail
source "$(dirname "$0")/env_homer.sh"

N_PER_HOP="${N_PER_HOP:-200}"   # 200 → 600 questions total (2/3/4 hops)
BATCH="${BATCH:-4}"

OUT_DIR="results/llama70b/musique"
CHAINS="$OUT_DIR/rewriting_chains_musique.csv"
F1="$OUT_DIR/rewriting_chains_musique_answer_f1.csv"

echo ""
echo "### 1/5 — Rewriting chains (Llama-3.1-70B-Instruct, 4-bit NF4)"
python scripts/llama70b/musique/rewriting_pipeline_musique.py \
  --n-per-hop "$N_PER_HOP" \
  --n-iterations 3 \
  --temperature 0.7 \
  --max-new-tokens 2048 \
  --batch-size "$BATCH" \
  --output "$CHAINS"

echo ""
echo "### 2/5 — Answer F1"
python scripts/llama70b/musique/answer_f1_eval_musique.py \
  --input "$CHAINS" \
  --output "$F1" \
  --batch-size "$BATCH" \
  --resume

echo ""
echo "### 3/5 — BERTScore (roberta-large)"
python scripts/llama70b/_common/bertscore_eval.py \
  --input "$CHAINS" --batch-size 32

echo ""
echo "### 4/5 — BLEURT (BLEURT-20) + bleurt_answer"
python scripts/llama70b/_common/bleurt_eval.py \
  --input "$CHAINS" --f1-csv "$F1" --batch-size 64

echo ""
echo "### 5/5 — Perplexity (Llama-3.1-70B 4-bit)"
python scripts/llama70b/_common/perplexity_eval.py \
  --input "$CHAINS"

echo ""
echo "### bonus — OpenFActScore (AFG=OLMo-2-7B-SFT, AFV=Gemma-3-4B)"
python scripts/llama70b/_common/openfactscore_eval.py \
  --input "$CHAINS" --topic-mode qid

echo ""
echo "DONE. Outputs under $OUT_DIR/"
