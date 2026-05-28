#!/bin/bash
# Quick smoke test on Homer: 1 question per dataset, full pipeline.
# Verifies that:
#  - OLMo-3.1-32B-Instruct loads in bf16 via vLLM (TP=2) on Homer's GPUs
#  - chain CSV is written
#  - all eval scripts run end-to-end
# Should finish in well under 30 min.
set -euo pipefail
source "$(dirname "$0")/env_homer.sh"

echo ""
echo "============================================================"
echo "  MuSiQue smoke (1 question, pilot 2-hop)"
echo "============================================================"
OUT="results/olmo32b/musique/rewriting_chains_musique_smoke.csv"
python scripts/olmo32b/musique/rewriting_pipeline_musique.py \
  --smoke-test --output "$OUT"
python scripts/olmo32b/musique/answer_f1_eval_musique.py \
  --input "$OUT"
python scripts/olmo32b/_common/bertscore_eval.py --input "$OUT" --batch-size 16
python scripts/olmo32b/_common/perplexity_eval.py --input "$OUT" --smoke-test

echo ""
echo "============================================================"
echo "  NewsQA smoke (1 story)"
echo "============================================================"
OUT="results/olmo32b/newsqa/rewriting_chains_newsqa_smoke.csv"
python scripts/olmo32b/newsqa/rewriting_pipeline_newsqa.py \
  --smoke-test --output "$OUT"
python scripts/olmo32b/newsqa/answer_f1_eval_newsqa.py \
  --input "$OUT" --smoke-test

echo ""
echo "============================================================"
echo "  FictionalQA smoke (1 fiction)"
echo "============================================================"
OUT="results/olmo32b/fictionalqa/rewriting_chains_fictionalqa_smoke.csv"
python scripts/olmo32b/fictionalqa/rewriting_pipeline_fictionalqa.py \
  --smoke-test --output "$OUT"
python scripts/olmo32b/fictionalqa/answer_f1_eval_fictionalqa.py \
  --input "$OUT" --smoke-test

echo ""
echo "SMOKE TEST OK"
