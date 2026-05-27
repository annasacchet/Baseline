#!/bin/bash
# FULL smoke test on MuSiQue (1 question, all 12 chains, end-to-end).
# Runs: rewriting -> Answer F1 -> BERTScore -> BLEURT (text + answer) ->
#       Perplexity -> OpenFActScore.
#
# Llama-3.1-70B is served via vLLM (AWQ-INT4, TP=2) for rewriting + F1.
# Perplexity uses HF transformers + bnb (vLLM doesn't expose log-likelihoods);
# OFS keeps OLMo-2-7B AFG + Gemma-3-4B AFV on HF transformers.
#
# Output goes to results/llama70b/musique/smoke/, separate from the full-run
# files so the smoke can be re-run without polluting real results.
#
# Usage (on Homer, inside tmux):
#   bash ~/Baseline/scripts/llama70b/launch/launch_smoke_full_musique_homer.sh \
#     2>&1 | tee ~/Baseline/logs/llama_smoke_full_musique.log
#
# Skip slow steps:
#   SKIP_PPL=1 bash launch_smoke_full_musique_homer.sh   # skip Perplexity (HF 70B is slow)
#   SKIP_OFS=1 bash launch_smoke_full_musique_homer.sh   # skip OpenFActScore
set -euo pipefail
source "$(dirname "$0")/env_homer.sh"

# Tell python to flush stdout/stderr immediately so the | tee log updates live.
export PYTHONUNBUFFERED=1

OUT_DIR="results/llama70b/musique/smoke"
mkdir -p "$OUT_DIR"

CHAINS="$OUT_DIR/rewriting_chains_musique_smoke.csv"
F1="$OUT_DIR/rewriting_chains_musique_smoke_answer_f1.csv"
BERT="$OUT_DIR/rewriting_chains_musique_smoke_bertscore.csv"
BLEURT="$OUT_DIR/rewriting_chains_musique_smoke_bleurt.csv"
PPL="$OUT_DIR/rewriting_chains_musique_smoke_perplexity.csv"

# vLLM tuning knobs (override via env if needed).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

echo ""
echo "############################################################"
echo "#  1/6 — Rewriting chains (Llama-3.1-70B-AWQ via vLLM)     #"
echo "############################################################"
python scripts/llama70b/musique/rewriting_pipeline_musique.py \
  --smoke-test \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-mem-util "$GPU_MEM_UTIL" \
  --output "$CHAINS"

echo ""
echo "############################################################"
echo "#  2/6 — Answer F1 (Llama-3.1-70B-AWQ via vLLM)            #"
echo "############################################################"
python scripts/llama70b/musique/answer_f1_eval_musique.py \
  --input "$CHAINS" \
  --output "$F1" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-mem-util "$GPU_MEM_UTIL"

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

if [ -z "${SKIP_PPL:-}" ]; then
  echo ""
  echo "############################################################"
  echo "#  5/6 — Perplexity (Llama-3.1-70B 4-bit via HF/bnb)       #"
  echo "############################################################"
  echo "  (skip with SKIP_PPL=1 — this step is slow on consumer GPUs)"
  python scripts/llama70b/_common/perplexity_eval.py \
    --input "$CHAINS" \
    --output "$PPL" \
    --smoke-test
else
  echo ""
  echo "[SKIP] 5/6 Perplexity (SKIP_PPL=1)"
fi

if [ -z "${SKIP_OFS:-}" ]; then
  echo ""
  echo "############################################################"
  echo "#  6/6 — OpenFActScore (AFG=OLMo-2-7B-SFT, AFV=Gemma-3-4B) #"
  echo "############################################################"
  # --limit 12 → score the first 12 step>0 rows of the chain set (3 steps ×
  # 4 instruction_types × 3 wordings = 36 rows). Bump/remove to score all.
  python scripts/llama70b/_common/openfactscore_eval.py \
    --input "$CHAINS" \
    --topic-mode qid \
    --limit 12
else
  echo ""
  echo "[SKIP] 6/6 OpenFActScore (SKIP_OFS=1)"
fi

echo ""
echo "############################################################"
echo "  DONE — smoke outputs in $OUT_DIR/"
echo "############################################################"
ls -lh "$OUT_DIR"
