#!/bin/bash
# Full FictionalQA pipeline (rewriting + 5 evals) on Homer with Llama-3.1-70B 4-bit.
set -euo pipefail
source "$(dirname "$0")/env_homer.sh"

N_FICTIONS="${N_FICTIONS:-100}"
BATCH="${BATCH:-4}"

OUT_DIR="results/llama70b/fictionalqa"
CHAINS="$OUT_DIR/rewriting_chains_fictionalqa.csv"
F1="$OUT_DIR/rewriting_chains_fictionalqa_answer_f1.csv"

echo ""
echo "### 1/5 — Rewriting chains (Llama-3.1-70B-Instruct, 4-bit NF4)"
python scripts/llama70b/fictionalqa/rewriting_pipeline_fictionalqa.py \
  --n-fictions "$N_FICTIONS" \
  --n-iterations 3 \
  --temperature 0.7 \
  --max-new-tokens 4096 \
  --batch-size "$BATCH" \
  --output "$CHAINS"

echo ""
echo "### 2/5 — Answer F1"
python scripts/llama70b/fictionalqa/answer_f1_eval_fictionalqa.py \
  --input "$CHAINS" --output "$F1" --batch-size "$BATCH"

echo ""
echo "### 3/5 — BERTScore"
python scripts/llama70b/_common/bertscore_eval.py --input "$CHAINS" --batch-size 32

echo ""
echo "### 4/5 — BLEURT + bleurt_answer"
python scripts/llama70b/_common/bleurt_eval.py \
  --input "$CHAINS" --f1-csv "$F1" --batch-size 64

echo ""
echo "### 5/5 — Perplexity (Llama-3.1-70B 4-bit)"
python scripts/llama70b/_common/perplexity_eval.py --input "$CHAINS"

echo ""
echo "### bonus — OpenFActScore (topic from first fiction line)"
python scripts/llama70b/_common/openfactscore_eval.py \
  --input "$CHAINS" --topic-mode first-line

echo ""
echo "DONE. Outputs under $OUT_DIR/"
