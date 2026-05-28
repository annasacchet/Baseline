#!/bin/bash
# Self-Refine (RQ3) pipeline on FictionalQA — Llama-3.1-70B-AWQ via vLLM on Homer.
set -euo pipefail
source "$(dirname "$0")/env_homer.sh"

N_FICTIONS="${N_FICTIONS:-100}"

OUT_DIR="results/llama70b/fictionalqa"
BASELINE="$OUT_DIR/rewriting_chains_fictionalqa.csv"
CHAINS="$OUT_DIR/self_refine_chains_fictionalqa.csv"
F1="$OUT_DIR/self_refine_chains_fictionalqa_answer_f1.csv"

if [ ! -f "$BASELINE" ]; then
  echo ""
  echo "### 0/6 — Baseline rewriting (needed for byte-identical E0)"
  python scripts/llama70b/fictionalqa/rewriting_pipeline_fictionalqa.py \
    --n-fictions "$N_FICTIONS" \
    --output "$BASELINE"
fi

echo ""
echo "### 1/6 — Self-refine chains (Rewriter/Critic/Refiner, Llama-70B-AWQ vLLM)"
python scripts/llama70b/fictionalqa/self_refine_pipeline_fictionalqa.py \
  --n-fictions "$N_FICTIONS" \
  --n-iterations 3 \
  --output "$CHAINS" \
  --baseline-csv "$BASELINE"

echo ""
echo "### 2/6 — Answer F1 on self-refine chains"
python scripts/llama70b/fictionalqa/answer_f1_eval_fictionalqa.py \
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
echo "### 6/6 — OpenFActScore (topic from first fiction line)"
python scripts/llama70b/_common/openfactscore_eval.py \
  --input "$CHAINS" --topic-mode first-line

echo ""
echo "DONE. Self-refine outputs in $OUT_DIR/"
