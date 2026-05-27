#!/bin/bash
# FULL smoke test on MuSiQue (1 question, all 12 chains, end-to-end).
# Runs: rewriting -> Answer F1 -> BERTScore -> BLEURT (text + answer) ->
#       Perplexity -> OpenFActScore.
#
# Output goes to results/llama70b/musique/smoke/, separate from the full-run
# files so the smoke can be re-run without polluting real results.
#
# Usage (on Homer, inside tmux):
#   bash ~/Baseline/scripts/llama70b/launch/launch_smoke_full_musique_homer.sh \
#     2>&1 | tee ~/Baseline/logs/llama_smoke_full_musique.log
set -euo pipefail
source "$(dirname "$0")/env_homer.sh"

OUT_DIR="results/llama70b/musique/smoke"
mkdir -p "$OUT_DIR"

CHAINS="$OUT_DIR/rewriting_chains_musique_smoke.csv"
F1="$OUT_DIR/rewriting_chains_musique_smoke_answer_f1.csv"
BERT="$OUT_DIR/rewriting_chains_musique_smoke_bertscore.csv"
BLEURT="$OUT_DIR/rewriting_chains_musique_smoke_bleurt.csv"
PPL="$OUT_DIR/rewriting_chains_musique_smoke_perplexity.csv"

# Small batches to keep memory bounded; everything still finishes in <1 h on
# Homer's 2x A6000 48GB.
BATCH="${BATCH:-2}"

echo ""
echo "############################################################"
echo "#  1/6 — Rewriting chains (Llama-3.1-70B-Instruct, 4-bit)  #"
echo "############################################################"
python scripts/llama70b/musique/rewriting_pipeline_musique.py \
  --smoke-test \
  --batch-size "$BATCH" \
  --output "$CHAINS"

echo ""
echo "############################################################"
echo "#  2/6 — Answer F1 (Llama-3.1-70B QA)                      #"
echo "############################################################"
python scripts/llama70b/musique/answer_f1_eval_musique.py \
  --input "$CHAINS" \
  --output "$F1" \
  --batch-size "$BATCH"

echo ""
echo "############################################################"
echo "#  3/6 — BERTScore (roberta-large, layer 17)               #"
echo "############################################################"
python scripts/llama70b/_common/bertscore_eval.py \
  --input "$CHAINS" \
  --output "$BERT" \
  --batch-size 16

echo ""
echo "############################################################"
echo "#  4/6 — BLEURT (text baseline + consecutive + answer)     #"
echo "############################################################"
python scripts/llama70b/_common/bleurt_eval.py \
  --input "$CHAINS" \
  --f1-csv "$F1" \
  --output "$BLEURT" \
  --batch-size 32 \
  --smoke-test

echo ""
echo "############################################################"
echo "#  5/6 — Perplexity (Llama-3.1-70B 4-bit)                  #"
echo "############################################################"
python scripts/llama70b/_common/perplexity_eval.py \
  --input "$CHAINS" \
  --output "$PPL" \
  --smoke-test

echo ""
echo "############################################################"
echo "#  6/6 — OpenFActScore (AFG=OLMo-2-7B-SFT, AFV=Gemma-3-4B) #"
echo "############################################################"
# --limit 12 → score every step>0 row of the single chain (3 steps × 4
# instruction_types × 3 wordings = 36 rows; --limit 12 stays under that to
# keep the smoke quick — bump or remove to score the whole chain set).
python scripts/llama70b/_common/openfactscore_eval.py \
  --input "$CHAINS" \
  --topic-mode qid \
  --limit 12

echo ""
echo "############################################################"
echo "  DONE — smoke outputs in $OUT_DIR/"
echo "############################################################"
ls -lh "$OUT_DIR"
