#!/bin/bash
# Shared FORWARD-pipeline driver for the final experiments.
#
# Runs, for one (model, dataset) pair, every metric EXCEPT factscore + recall:
#   1. rewriting (iterative, 3 steps)  — also emits n_tokens (token length)
#   2. Answer F1
#   3. BERTScore (Baseline sim(E_t,E_0) + Consecutive sim(E_t,E_{t-1}))
#   4. BLEURT    (baseline + consecutive + answer)
#   5. Perplexity
#
# Every step is RESUMABLE: rewriting skips chains already in the CSV, F1 /
# perplexity skip rows already scored, BERTScore/BLEURT recompute (cheap). Re-run
# the same launcher after an interruption and it continues from where it stopped.
#
# A launcher sources this file, sets the variables below, then calls
# `run_forward`. OpenFActScore + recall are NOT run here — see lib_ofs.sh.
#
# Required variables (set by the caller before calling run_forward):
#   MODEL_DIR     scripts/<model>            (e.g. olmo32b, qwen30b, llama70b, gptoss120b)
#   DATASET       musique | newsqa | fictionalqa
#   OUT_DIR       where the CSVs go
#   TAG           filename stem (e.g. musique_600q)
# Optional (sane defaults below; override via env):
#   N_PER_HOP (musique, default 200 -> 600q), N_ITEMS (newsqa/fictionalqa, default 600)
#   N_ITERATIONS (3), TEMPERATURE (0.7), SEED (42)
#   MAX_NEW_TOKENS, MAX_MODEL_LEN, GPU_MEM_UTIL, TP (tensor-parallel-size)
#   BACKEND (vllm|hf), QUANT (passed as --quantization if set)
#   EXTRA_REWRITE_ARGS / EXTRA_F1_ARGS  (raw extra CLI args)
#   SKIP_PPL=1, USE_4BIT_PPL=1
set -euo pipefail

# _supports_flag <script.py> <--flag>: true if the script's argparse --help
# advertises that flag. Lets us pass each optional flag ONLY to scripts that
# accept it, instead of hardcoding a fragile per-script matrix (the rewriting /
# answer-F1 CLIs differ a lot across model+dataset). --help output is cached on
# disk per script (argparse prints --help before any heavy import, so this is
# cheap). Written to be portable down to bash 3.2 (no associative arrays / -n).
_HELP_CACHE_DIR="${TMPDIR:-/tmp}/final_exp_help_cache.$$"
mkdir -p "$_HELP_CACHE_DIR"
_supports_flag() {
  local script="$1" flag="$2"
  local key; key="$(printf '%s' "$script" | tr '/.' '__')"
  local cache="$_HELP_CACHE_DIR/$key"
  [ -f "$cache" ] || python "$script" --help >"$cache" 2>&1 || true
  grep -q -- "$flag" "$cache"
}

run_forward() {
  : "${MODEL_DIR:?set MODEL_DIR}"
  : "${DATASET:?set DATASET}"
  : "${OUT_DIR:?set OUT_DIR}"
  : "${TAG:?set TAG}"

  local N_PER_HOP="${N_PER_HOP:-200}"
  local N_ITEMS="${N_ITEMS:-600}"
  local N_ITERATIONS="${N_ITERATIONS:-3}"
  local TEMPERATURE="${TEMPERATURE:-0.7}"
  local SEED="${SEED:-42}"
  local MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
  local MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
  local GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
  local TP="${TP:-}"
  # ${BACKEND-vllm}: only an UNSET BACKEND falls back to vllm; an explicitly
  # exported empty BACKEND="" (llama musique = AWQ-only, no --backend flag) is
  # preserved as empty.
  local BACKEND="${BACKEND-vllm}"
  local QUANT="${QUANT:-}"

  mkdir -p "$OUT_DIR" logs

  local PY="scripts/${MODEL_DIR}"
  local CHAINS="$OUT_DIR/rewriting_chains_${TAG}.csv"
  local F1="$OUT_DIR/rewriting_chains_${TAG}_answer_f1.csv"
  local BERT="$OUT_DIR/rewriting_chains_${TAG}_bertscore.csv"
  local BLEURT="$OUT_DIR/rewriting_chains_${TAG}_bleurt.csv"
  local PPL="$OUT_DIR/rewriting_chains_${TAG}_perplexity.csv"

  # The per-(model,dataset) rewriting and Answer-F1 CLIs are heterogeneous (vLLM
  # vs HF/bnb 4-bit; --backend present or not; --resume present or not). Rather
  # than hardcode that matrix, each generation flag is added below ONLY if the
  # target script advertises it (_add_if_supported / _supports_flag).
  #
  # BACKEND semantics: "hf" -> pass --backend hf (+ --enforce-eager), "vllm" ->
  # pass --backend vllm, "" (empty, e.g. llama musique AWQ-only) -> never pass
  # --backend. Empty is preserved because the launchers export BACKEND="".

  echo ""
  echo "============================================================"
  echo "  FORWARD PIPELINE — ${MODEL_DIR} / ${DATASET}"
  echo "============================================================"
  echo "  TAG            = $TAG"
  echo "  OUT_DIR        = $OUT_DIR"
  echo "  N_PER_HOP      = $N_PER_HOP   (musique: 200/hop = 600q)"
  echo "  N_ITEMS        = $N_ITEMS     (newsqa/fictionalqa target sample)"
  echo "  N_ITERATIONS   = $N_ITERATIONS"
  echo "  MAX_NEW_TOKENS = $MAX_NEW_TOKENS   MAX_MODEL_LEN = $MAX_MODEL_LEN"
  echo "  BACKEND        = ${BACKEND:-<pipeline default>}   QUANT = ${QUANT:-none}   TP = ${TP:-auto}"
  echo "============================================================"

  # ---- 1/5 rewriting ------------------------------------------------------- #
  # Per-dataset the sampling flag differs: musique=--n-per-hop, newsqa=--n-questions,
  # fictionalqa=--n-fictions. Per-style the generation flags differ (see above).
  echo ""
  echo "########## 1/5 — Rewriting chains (${MODEL_DIR}, ${DATASET}) ##########"
  local REWRITE_PY SAMPLE_ARGS
  case "$DATASET" in
    musique)     REWRITE_PY="$PY/musique/rewriting_pipeline_musique.py";       SAMPLE_ARGS=(--n-per-hop "$N_PER_HOP") ;;
    newsqa)      REWRITE_PY="$PY/newsqa/rewriting_pipeline_newsqa.py";          SAMPLE_ARGS=(--n-questions "$N_ITEMS") ;;
    fictionalqa) REWRITE_PY="$PY/fictionalqa/rewriting_pipeline_fictionalqa.py"; SAMPLE_ARGS=(--n-fictions "$N_ITEMS") ;;
    *) echo "[ERROR] unknown DATASET=$DATASET" >&2; return 1 ;;
  esac

  # Build the rewriting args by introspection: every flag is added only if this
  # particular pipeline accepts it. This absorbs all the vllm/hfbnb differences
  # (llama newsqa/fictionalqa lack --max-model-len etc.; some have --batch-size).
  local RW_ARGS=("${SAMPLE_ARGS[@]}" --n-iterations "$N_ITERATIONS"
                 --temperature "$TEMPERATURE" --max-new-tokens "$MAX_NEW_TOKENS"
                 --seed "$SEED" --output "$CHAINS")
  _supports_flag "$REWRITE_PY" --max-model-len && RW_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
  _supports_flag "$REWRITE_PY" --gpu-mem-util  && RW_ARGS+=(--gpu-mem-util "$GPU_MEM_UTIL")
  [ -n "$TP" ]    && _supports_flag "$REWRITE_PY" --tensor-parallel-size && RW_ARGS+=(--tensor-parallel-size "$TP")
  [ -n "$QUANT" ] && _supports_flag "$REWRITE_PY" --quantization         && RW_ARGS+=(--quantization "$QUANT")
  case "$BACKEND" in
    hf)   _supports_flag "$REWRITE_PY" --backend && RW_ARGS+=(--backend hf)
          _supports_flag "$REWRITE_PY" --enforce-eager && RW_ARGS+=(--enforce-eager) ;;
    vllm) _supports_flag "$REWRITE_PY" --backend && RW_ARGS+=(--backend vllm) ;;
  esac
  _supports_flag "$REWRITE_PY" --batch-size && RW_ARGS+=(--batch-size "${REWRITE_BATCH_SIZE:-4}")
  python "$REWRITE_PY" "${RW_ARGS[@]}" ${EXTRA_REWRITE_ARGS:-}

  # ---- 2/5 Answer F1 ------------------------------------------------------- #
  # The Answer-F1 CLIs differ a lot across model/dataset (vLLM vs HF/bnb 4-bit;
  # some have --resume/--backend, some only --batch-size). Same trick: pass each
  # flag only if the script advertises it.
  local F1_PY="$PY/${DATASET}/answer_f1_eval_${DATASET}.py"
  echo ""
  echo "########## 2/5 — Answer F1 (${MODEL_DIR}, ${DATASET}) ##########"
  local F1_ARGS=(--input "$CHAINS" --output "$F1")
  _supports_flag "$F1_PY" --max-model-len && F1_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
  _supports_flag "$F1_PY" --gpu-mem-util  && F1_ARGS+=(--gpu-mem-util "$GPU_MEM_UTIL")
  [ -n "$TP" ]    && _supports_flag "$F1_PY" --tensor-parallel-size && F1_ARGS+=(--tensor-parallel-size "$TP")
  [ -n "$QUANT" ] && _supports_flag "$F1_PY" --quantization         && F1_ARGS+=(--quantization "$QUANT")
  case "$BACKEND" in
    hf)   _supports_flag "$F1_PY" --backend && F1_ARGS+=(--backend hf)
          _supports_flag "$F1_PY" --enforce-eager && F1_ARGS+=(--enforce-eager) ;;
    vllm) _supports_flag "$F1_PY" --backend && F1_ARGS+=(--backend vllm) ;;
  esac
  _supports_flag "$F1_PY" --batch-size && F1_ARGS+=(--batch-size "${F1_BATCH_SIZE:-2}")
  _supports_flag "$F1_PY" --resume     && F1_ARGS+=(--resume)
  python "$F1_PY" "${F1_ARGS[@]}" ${EXTRA_F1_ARGS:-}

  # ---- 3/5 BERTScore (baseline + consecutive) ------------------------------ #
  echo ""
  echo "########## 3/5 — BERTScore (Baseline + Consecutive) ##########"
  python "$PY/_common/bertscore_eval.py" \
    --input "$CHAINS" --output "$BERT" --batch-size 32

  # ---- 4/5 BLEURT (baseline + consecutive + answer) ------------------------ #
  echo ""
  echo "########## 4/5 — BLEURT (baseline + consecutive + answer) ##########"
  python "$PY/_common/bleurt_eval.py" \
    --input "$CHAINS" --f1-csv "$F1" --output "$BLEURT" --batch-size 64

  # ---- 5/5 Perplexity ------------------------------------------------------ #
  if [ -z "${SKIP_PPL:-}" ]; then
    echo ""
    echo "########## 5/5 — Perplexity ##########"
    local PPL_ARGS=()
    [ -n "${USE_4BIT_PPL:-}" ] && PPL_ARGS+=(--use-4bit)
    python "$PY/_common/perplexity_eval.py" \
      --input "$CHAINS" --output "$PPL" ${PPL_ARGS[@]+"${PPL_ARGS[@]}"}
  else
    echo ""
    echo "[SKIP] 5/5 Perplexity (SKIP_PPL=1)"
  fi

  echo ""
  echo "############################################################"
  echo "  FORWARD DONE — outputs in $OUT_DIR/"
  echo "  (token length lives in the chains CSV as the n_tokens column)"
  echo "############################################################"
  ls -lh "$OUT_DIR" 2>/dev/null || true
}
