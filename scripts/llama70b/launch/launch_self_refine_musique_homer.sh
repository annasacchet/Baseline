#!/bin/bash
# Self-Refine (RQ3) pipeline on MuSiQue — Llama-3.1-70B-AWQ via vLLM on Homer.
#
# Order of operations:
#   1. If the baseline rewriting CSV is missing, run rewriting first (so E0 is
#      byte-identical between RQ1 and RQ3).
#   2. Self-refine chains (Rewriter → Critic → Refiner, batched).
#   3. Answer F1 + BERTScore + BLEURT + Perplexity + OFS on the self-refine
#      CSV. The eval scripts only read the standard chain columns, so the
#      extra draft_text / critic_feedback columns are ignored harmlessly.
#
# Usage:
#   tmux new -s llama_self_refine_musique
#   bash scripts/llama70b/launch/launch_self_refine_musique_homer.sh \
#     2>&1 | tee logs/llama_self_refine_musique.log
set -euo pipefail
source "$(dirname "$0")/env_homer.sh"

N_PER_HOP="${N_PER_HOP:-200}"

OUT_DIR="results/llama70b/musique"
BASELINE="$OUT_DIR/rewriting_chains_musique.csv"
CHAINS="$OUT_DIR/self_refine_chains_musique.csv"
F1="$OUT_DIR/self_refine_chains_musique_answer_f1.csv"

if [ ! -f "$BASELINE" ]; then
  echo ""
  echo "### 0/6 — Baseline rewriting (needed for byte-identical E0)"
  python scripts/llama70b/musique/rewriting_pipeline_musique.py \
    --n-per-hop "$N_PER_HOP" \
    --n-iterations 3 \
    --temperature 0.7 \
    --max-new-tokens 2048 \
    --output "$BASELINE"
fi

echo ""
echo "### 1/6 — Self-refine chains (Rewriter/Critic/Refiner, Llama-70B-AWQ)"
python scripts/llama70b/musique/self_refine_pipeline_musique.py \
  --n-per-hop "$N_PER_HOP" \
  --n-iterations 3 \
  --output "$CHAINS" \
  --baseline-csv "$BASELINE"

echo ""
echo "### 2/6 — Answer F1 on self-refine chains"
python scripts/llama70b/musique/answer_f1_eval_musique.py \
  --input "$CHAINS" --output "$F1" --resume

echo ""
echo "### 3/6 — BERTScore (roberta-large)"
python scripts/llama70b/_common/bertscore_eval.py \
  --input "$CHAINS" --batch-size 32

echo ""
echo "### 4/6 — BLEURT (BLEURT-20) + bleurt_answer"
python scripts/llama70b/_common/bleurt_eval.py \
  --input "$CHAINS" --f1-csv "$F1" --batch-size 64

echo ""
echo "### 5/6 — Perplexity (Llama-3.1-70B 4-bit)"
python scripts/llama70b/_common/perplexity_eval.py --input "$CHAINS"

echo ""
echo "### 6/6 — OpenFActScore (AFG=OLMo-2-7B-SFT, AFV=Gemma-3-4B)"
python scripts/llama70b/_common/openfactscore_eval.py \
  --input "$CHAINS" --topic-mode qid

echo ""
echo "DONE. Self-refine outputs in $OUT_DIR/"
