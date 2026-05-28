#!/bin/bash
# Self-Refine (RQ3) pipeline on MuSiQue — OLMo-3.1-32B-Instruct via vLLM on Homer.
#
# Order of operations:
#   1. If the baseline rewriting CSV is missing, run rewriting first (so E0 is
#      byte-identical between RQ1 and RQ3).
#   2. Self-refine chains (Rewriter → Critic → Refiner, batched via vLLM).
#   3. Answer F1 + BERTScore + BLEURT + Perplexity + OFS on the self-refine
#      CSV. The eval scripts only read the standard chain columns, so the
#      extra draft_text / critic_feedback columns are ignored harmlessly.
#
# Usage:
#   tmux new -s olmo_self_refine_musique
#   bash scripts/olmo32b/launch/launch_self_refine_musique_homer.sh \
#     2>&1 | tee logs/olmo_self_refine_musique.log
set -euo pipefail
source "$(dirname "$0")/env_homer.sh"

N_PER_HOP="${N_PER_HOP:-200}"

OUT_DIR="results/olmo32b/musique"
BASELINE="$OUT_DIR/rewriting_chains_musique.csv"
CHAINS="$OUT_DIR/self_refine_chains_musique.csv"
F1="$OUT_DIR/self_refine_chains_musique_answer_f1.csv"

if [ ! -f "$BASELINE" ]; then
  echo ""
  echo "### 0/6 — Baseline rewriting (needed for byte-identical E0)"
  python scripts/olmo32b/musique/rewriting_pipeline_musique.py \
    --n-per-hop "$N_PER_HOP" \
    --n-iterations 3 \
    --temperature 0.7 \
    --max-new-tokens 2048 \
    --output "$BASELINE"
fi

echo ""
echo "### 1/6 — Self-refine chains (Rewriter/Critic/Refiner, OLMo-3.1-32B vLLM)"
python scripts/olmo32b/musique/self_refine_pipeline_musique.py \
  --n-per-hop "$N_PER_HOP" \
  --n-iterations 3 \
  --output "$CHAINS" \
  --baseline-csv "$BASELINE"

echo ""
echo "### 2/6 — Answer F1 on self-refine chains"
python scripts/olmo32b/musique/answer_f1_eval_musique.py \
  --input "$CHAINS" --output "$F1" --resume

echo ""
echo "### 3/6 — BERTScore (roberta-large)"
python scripts/olmo32b/_common/bertscore_eval.py \
  --input "$CHAINS" --batch-size 32

echo ""
echo "### 4/6 — BLEURT (BLEURT-20) + bleurt_answer"
python scripts/olmo32b/_common/bleurt_eval.py \
  --input "$CHAINS" --f1-csv "$F1" --batch-size 64

echo ""
echo "### 5/6 — Perplexity (OLMo-3.1-32B bf16)"
python scripts/olmo32b/_common/perplexity_eval.py --input "$CHAINS"

echo ""
echo "### 6/6 — OpenFActScore (AFG=OLMo-2-7B-SFT, AFV=Gemma-3-4B)"
python scripts/olmo32b/_common/openfactscore_eval.py \
  --input "$CHAINS" --topic-mode qid

echo ""
echo "DONE. Self-refine outputs in $OUT_DIR/"
