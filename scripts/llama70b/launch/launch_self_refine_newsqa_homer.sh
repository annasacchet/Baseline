#!/bin/bash
# Self-Refine (RQ3) pipeline on NewsQA — Llama-3.1-70B-AWQ via vLLM on Homer.
set -euo pipefail
source "$(dirname "$0")/env_homer.sh"

N_QUESTIONS="${N_QUESTIONS:-100}"

OUT_DIR="results/llama70b/newsqa"
BASELINE="$OUT_DIR/rewriting_chains_newsqa.csv"
CHAINS="$OUT_DIR/self_refine_chains_newsqa.csv"
F1="$OUT_DIR/self_refine_chains_newsqa_answer_f1_span.csv"

if [ ! -f "$BASELINE" ]; then
  echo ""
  echo "### 0/6 — Baseline rewriting (needed for byte-identical E0)"
  python scripts/llama70b/newsqa/rewriting_pipeline_newsqa.py \
    --n-questions "$N_QUESTIONS" \
    --output "$BASELINE"
fi

echo ""
echo "### 1/6 — Self-refine chains (Rewriter/Critic/Refiner, Llama-70B-AWQ vLLM)"
python scripts/llama70b/newsqa/self_refine_pipeline_newsqa.py \
  --n-questions "$N_QUESTIONS" \
  --n-iterations 3 \
  --output "$CHAINS" \
  --baseline-csv "$BASELINE"

echo ""
echo "### 2/6 — Answer F1 (extractive span prompt) on self-refine chains"
python scripts/llama70b/newsqa/answer_f1_eval_newsqa.py \
  --input "$CHAINS" --output "$F1"

echo ""
echo "### 3/6 — BERTScore"
python scripts/llama70b/_common/bertscore_eval.py --input "$CHAINS" --batch-size 32

echo ""
echo "### 4/6 — BLEURT + bleurt_answer"
python scripts/llama70b/_common/bleurt_eval.py \
  --input "$CHAINS" --f1-csv "$F1" --batch-size 64

echo ""
echo "### 5/6 — Perplexity (Llama-3.1-70B 4-bit)"
python scripts/llama70b/_common/perplexity_eval.py --input "$CHAINS"

echo ""
echo "### 6/6 — OpenFActScore (topic from first article line)"
python scripts/llama70b/_common/openfactscore_eval.py \
  --input "$CHAINS" --topic-mode first-line

echo ""
echo "DONE. Self-refine outputs in $OUT_DIR/"
