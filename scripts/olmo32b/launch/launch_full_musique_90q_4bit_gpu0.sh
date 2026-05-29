#!/bin/bash
# Full MuSiQue 90q run on a SINGLE GPU (GPU 1) with OLMo-3.1-32B-Instruct
# quantized 4-bit (NF4 / bitsandbytes). Balanced 30/30/30 over 2/3/4-hop.
#
# Mirrors scripts/llama70b/launch/launch_full_musique_homer.sh but:
#   * 90 questions (N_PER_HOP=30) instead of 300/600
#   * 4-bit quantization (vLLM bnb for rewriting+F1, HF bnb for perplexity)
#     so the 32B fits on one consumer GPU
#   * forced onto GPU 1 via CUDA_VISIBLE_DEVICES=1, tensor_parallel_size=1
#
# Runs: rewriting -> Answer F1 -> BERTScore -> BLEURT -> Perplexity.
# OpenFActScore is SKIPPED here — run it separately in its own env, manually,
# as the last step.
#
# Usage:
#   tmux new -s olmo_90q
#   bash scripts/olmo32b/launch/launch_full_musique_90q_4bit_gpu0.sh \
#       2>&1 | tee logs/olmo_musique_90q.log
#
# Overridable via env:
#   N_PER_HOP (30), MAX_NEW_TOKENS (2048), MAX_MODEL_LEN (8192),
#   GPU_MEM_UTIL (0.90), TEMPERATURE (0.7), SEED (42), GPU (1),
#   OUT_DIR (results/olmo32b/musique/90q), SKIP_PPL=1
# ----------------------------------------------------------------------------
set -euo pipefail

# Optional shared env (HF cache on NAS + dataset paths). If env_homer.sh is
# present we source it; otherwise we fall back to sane defaults so the script
# runs standalone. Any variable you already exported in your shell wins.
_ENV_HOMER="$(dirname "$0")/env_homer.sh"
if [ -f "$_ENV_HOMER" ]; then
  source "$_ENV_HOMER"
else
  echo "[env] env_homer.sh not found — using defaults (override via shell exports)"
  export HF_HOME="${HF_HOME:-/mnt/dmif-nas/mitel/sacchet/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/dmif-nas/mitel/sacchet/hf_cache/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/dmif-nas/mitel/sacchet/hf_cache}"
  export MUSIQUE_DATASET="${MUSIQUE_DATASET:-/home/sacchet/Baseline/musique_ans_v1.0_dev.jsonl}"
  echo "[env] HF_HOME=$HF_HOME"
  echo "[env] MUSIQUE_DATASET=$MUSIQUE_DATASET"
fi
export PYTHONUNBUFFERED=1

# --- force a single GPU --------------------------------------------------- #
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[GPU] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (tensor_parallel_size=1)"

# ----------------------------------------------------------------------------
N_PER_HOP="${N_PER_HOP:-30}"
TOTAL_Q=$((N_PER_HOP * 3))
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
QUANT="bitsandbytes"        # NF4 4-bit (vLLM on-the-fly, or HF bnb)
BACKEND="${BACKEND:-vllm}"  # 'vllm' = ~10x faster. FlashInfer JIT (needs nvcc>=12)
                            # is avoided via VLLM_ATTENTION_BACKEND=FLASH_ATTN +
                            # --enforce-eager (set below). 'hf' = transformers
                            # 4-bit fallback (slow but no compilation).
# Force the precompiled FlashAttention backend so vLLM never JIT-builds a kernel
# with the old system nvcc (11.5). Validated on Homer: load + generate work.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
OUT_DIR="${OUT_DIR:-results/olmo32b/musique/${TOTAL_Q}q}"

mkdir -p "$OUT_DIR" logs

CHAINS="$OUT_DIR/rewriting_chains_musique_${TOTAL_Q}q.csv"
F1="$OUT_DIR/rewriting_chains_musique_${TOTAL_Q}q_answer_f1.csv"
BERT="$OUT_DIR/rewriting_chains_musique_${TOTAL_Q}q_bertscore.csv"
BLEURT="$OUT_DIR/rewriting_chains_musique_${TOTAL_Q}q_bleurt.csv"
PPL="$OUT_DIR/rewriting_chains_musique_${TOTAL_Q}q_perplexity.csv"

echo ""
echo "============================================================"
echo "  OLMo-3.1-32B-Instruct (4-bit) MuSiQue ${TOTAL_Q}q — single GPU"
echo "============================================================"
echo "  N_PER_HOP        = $N_PER_HOP   (total: $TOTAL_Q questions, balanced 2/3/4-hop)"
echo "  BACKEND          = $BACKEND"
echo "  QUANTIZATION     = $QUANT (NF4 4-bit)"
echo "  GPU              = $GPU  (TP=1)"
echo "  MAX_NEW_TOKENS   = $MAX_NEW_TOKENS"
echo "  MAX_MODEL_LEN    = $MAX_MODEL_LEN"
echo "  GPU_MEM_UTIL     = $GPU_MEM_UTIL"
echo "  TEMPERATURE      = $TEMPERATURE"
echo "  SEED             = $SEED"
echo "  OUT_DIR          = $OUT_DIR"
echo "============================================================"

echo ""
echo "############################################################"
echo "#  1/5 — Rewriting chains (OLMo-3.1-32B 4-bit, $BACKEND)        #"
echo "############################################################"
python scripts/olmo32b/musique/rewriting_pipeline_musique.py \
  --n-per-hop "$N_PER_HOP" \
  --n-iterations 3 \
  --temperature "$TEMPERATURE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-mem-util "$GPU_MEM_UTIL" \
  --tensor-parallel-size 1 \
  --quantization "$QUANT" \
  --backend "$BACKEND" \
  --enforce-eager \
  --seed "$SEED" \
  --output "$CHAINS"

echo ""
echo "############################################################"
echo "#  2/5 — Answer F1 (OLMo-3.1-32B 4-bit, $BACKEND)              #"
echo "############################################################"
python scripts/olmo32b/musique/answer_f1_eval_musique.py \
  --input "$CHAINS" \
  --output "$F1" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-mem-util "$GPU_MEM_UTIL" \
  --tensor-parallel-size 1 \
  --quantization "$QUANT" \
  --backend "$BACKEND" \
  --enforce-eager \
  --resume

echo ""
echo "############################################################"
echo "#  3/5 — BERTScore (roberta-large)                        #"
echo "############################################################"
python scripts/olmo32b/_common/bertscore_eval.py \
  --input "$CHAINS" --output "$BERT" --batch-size 32

echo ""
echo "############################################################"
echo "#  4/5 — BLEURT (BLEURT-20) + bleurt_answer               #"
echo "############################################################"
python scripts/olmo32b/_common/bleurt_eval.py \
  --input "$CHAINS" --f1-csv "$F1" --output "$BLEURT" --batch-size 64

if [ -z "${SKIP_PPL:-}" ]; then
  echo ""
  echo "############################################################"
  echo "#  5/5 — Perplexity (OLMo-3.1-32B 4-bit via HF/bnb)       #"
  echo "############################################################"
  python scripts/olmo32b/_common/perplexity_eval.py \
    --input "$CHAINS" --output "$PPL" --use-4bit
else
  echo ""
  echo "[SKIP] 5/5 Perplexity (SKIP_PPL=1)"
fi

echo ""
echo "############################################################"
echo "  DONE — ${TOTAL_Q}q outputs in $OUT_DIR/"
echo "############################################################"
ls -lh "$OUT_DIR"
echo ""
echo "OpenFActScore va lanciato a parte nel suo env, come ultimo step:"
echo "  python scripts/olmo32b/_common/openfactscore_eval.py \\"
echo "    --input $CHAINS --topic-mode qid"
