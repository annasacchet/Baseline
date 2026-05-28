#!/bin/bash
# Full NewsQA pipeline (rewriting + 5 evals) on Homer with OLMo-3.1-32B bf16.
set -euo pipefail
source "$(dirname "$0")/env_homer.sh"

N_QUESTIONS="${N_QUESTIONS:-100}"

OUT_DIR="results/olmo32b/newsqa"
CHAINS="$OUT_DIR/rewriting_chains_newsqa.csv"
F1="$OUT_DIR/rewriting_chains_newsqa_answer_f1_span.csv"

echo ""
echo "### 1/5 — Rewriting chains (OLMo-3.1-32B-Instruct, bf16 via vLLM)"
python scripts/olmo32b/newsqa/rewriting_pipeline_newsqa.py \
  --n-questions "$N_QUESTIONS" \
  --n-iterations 3 \
  --temperature 0.7 \
  --max-new-tokens 4096 \
  --output "$CHAINS"

echo ""
echo "### 2/5 — Answer F1 (extractive span prompt)"
python scripts/olmo32b/newsqa/answer_f1_eval_newsqa.py \
  --input "$CHAINS" --output "$F1"

echo ""
echo "### 3/5 — BERTScore"
python scripts/olmo32b/_common/bertscore_eval.py --input "$CHAINS" --batch-size 32

echo ""
echo "### 4/5 — BLEURT + bleurt_answer"
python scripts/olmo32b/_common/bleurt_eval.py \
  --input "$CHAINS" --f1-csv "$F1" --batch-size 64

echo ""
echo "### 5/5 — Perplexity (OLMo-3.1-32B bf16)"
python scripts/olmo32b/_common/perplexity_eval.py --input "$CHAINS"

echo ""
echo "### bonus — OpenFActScore (topic from first article line)"
python scripts/olmo32b/_common/openfactscore_eval.py \
  --input "$CHAINS" --topic-mode first-line

echo ""
echo "DONE. Outputs under $OUT_DIR/"
