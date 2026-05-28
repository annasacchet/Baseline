#!/bin/bash
# Full MuSiQue pipeline on Homer with Llama-3.1-70B-AWQ via vLLM.
# Runs: rewriting → Answer F1 → BERTScore → BLEURT → Perplexity.
# OpenFActScore is SKIPPED here — run it separately from the pipe311 env.
#
# Llama-3.1-70B-AWQ-INT4 is served via vLLM (TP=2) inside conda env `vllm311`.
# Perplexity uses HF transformers + bnb 4-bit (still in vllm311; vLLM doesn't
# expose log-likelihoods).
#
# ----------------------------------------------------------------------------
# Parametri configurabili da CLI / env (sovrascrivi via N_PER_HOP=… ecc.)
# ----------------------------------------------------------------------------
#   N_PER_HOP        = questions per hop class (2/3/4-hop). Default 100 → 300 q.
#   MAX_NEW_TOKENS   = max output tokens per rewrite. Default 3500 (riduce
#                      i troncamenti visti nello smoke).
#   MAX_MODEL_LEN    = vLLM context window. Default 8192.
#   GPU_MEM_UTIL     = frazione VRAM usata da vLLM. Default 0.90.
#   TEMPERATURE      = rewriting sampling temperature. Default 0.7.
#   SEED             = random seed per sampling questions. Default 42.
#   OUT_DIR          = dove finiscono i CSV. Default results/llama70b/musique/<N_total>q
#   SKIP_PPL=1       = salta lo step 5/5 (Perplexity)
#
# Esempi:
#   bash launch_full_musique_homer.sh                       # default: 300 q (100/100/100)
#   N_PER_HOP=50 bash launch_full_musique_homer.sh          # 150 q
#   N_PER_HOP=200 MAX_NEW_TOKENS=4096 bash launch_full_musique_homer.sh
#   SKIP_PPL=1 bash launch_full_musique_homer.sh            # senza perplexity
# ----------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/env_homer.sh"
export PYTHONUNBUFFERED=1

# Use the vllm311 conda env if present
if command -v conda >/dev/null 2>&1; then
  CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  [ -f "$CONDA_SH" ] && source "$CONDA_SH"
  conda activate vllm311 2>/dev/null || echo "[WARN] couldn't activate vllm311, using current env" >&2
fi
echo "Active python: $(command -v python) ($(python --version 2>&1))"

# ----------------------------------------------------------------------------
N_PER_HOP="${N_PER_HOP:-100}"
TOTAL_Q=$((N_PER_HOP * 3))
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3500}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-results/llama70b/musique/${TOTAL_Q}q}"

mkdir -p "$OUT_DIR"

CHAINS="$OUT_DIR/rewriting_chains_musique_${TOTAL_Q}q.csv"
F1="$OUT_DIR/rewriting_chains_musique_${TOTAL_Q}q_answer_f1.csv"
BERT="$OUT_DIR/rewriting_chains_musique_${TOTAL_Q}q_bertscore.csv"
BLEURT="$OUT_DIR/rewriting_chains_musique_${TOTAL_Q}q_bleurt.csv"
PPL="$OUT_DIR/rewriting_chains_musique_${TOTAL_Q}q_perplexity.csv"

echo ""
echo "============================================================"
echo "  Llama-3.1-70B-AWQ MuSiQue full run"
echo "============================================================"
echo "  N_PER_HOP        = $N_PER_HOP    (total: $TOTAL_Q questions, balanced 2/3/4-hop)"
echo "  MAX_NEW_TOKENS   = $MAX_NEW_TOKENS"
echo "  MAX_MODEL_LEN    = $MAX_MODEL_LEN"
echo "  GPU_MEM_UTIL     = $GPU_MEM_UTIL"
echo "  TEMPERATURE      = $TEMPERATURE"
echo "  SEED             = $SEED"
echo "  OUT_DIR          = $OUT_DIR"
echo "  CHAINS           = $CHAINS"
echo "============================================================"

echo ""
echo "############################################################"
echo "#  1/5 — Rewriting chains (Llama-3.1-70B-AWQ via vLLM)     #"
echo "############################################################"
python scripts/llama70b/musique/rewriting_pipeline_musique.py \
  --n-per-hop "$N_PER_HOP" \
  --n-iterations 3 \
  --temperature "$TEMPERATURE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-mem-util "$GPU_MEM_UTIL" \
  --seed "$SEED" \
  --output "$CHAINS"

echo ""
echo "############################################################"
echo "#  2/5 — Answer F1 (Llama-3.1-70B-AWQ via vLLM)            #"
echo "############################################################"
python scripts/llama70b/musique/answer_f1_eval_musique.py \
  --input "$CHAINS" \
  --output "$F1" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-mem-util "$GPU_MEM_UTIL" \
  --resume

echo ""
echo "############################################################"
echo "#  3/5 — BERTScore (roberta-large, layer 17)               #"
echo "############################################################"
python scripts/llama70b/_common/bertscore_eval.py \
  --input "$CHAINS" \
  --output "$BERT" \
  --batch-size 32

echo ""
echo "############################################################"
echo "#  4/5 — BLEURT (baseline + consecutive + answer)          #"
echo "############################################################"
python scripts/llama70b/_common/bleurt_eval.py \
  --input "$CHAINS" \
  --f1-csv "$F1" \
  --output "$BLEURT" \
  --batch-size 64

if [ -z "${SKIP_PPL:-}" ]; then
  echo ""
  echo "############################################################"
  echo "#  5/5 — Perplexity (Llama-3.1-70B 4-bit via HF/bnb)       #"
  echo "############################################################"
  echo "  (skip with SKIP_PPL=1 — this step is slow on consumer GPUs)"
  python scripts/llama70b/_common/perplexity_eval.py \
    --input "$CHAINS" \
    --output "$PPL"
else
  echo ""
  echo "[SKIP] 5/5 Perplexity (SKIP_PPL=1)"
fi

echo ""
echo "############################################################"
echo "  DONE — full-run outputs in $OUT_DIR/"
echo "############################################################"
ls -lh "$OUT_DIR"
echo ""
echo "OFS è da lanciare a parte nell'env pipe311:"
echo "  conda activate pipe311"
echo "  export TORCHDYNAMO_DISABLE=1"
echo "  python scripts/llama70b/_common/openfactscore_eval.py \\"
echo "    --input $CHAINS \\"
echo "    --topic-mode qid"
