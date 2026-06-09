#!/bin/bash
# Shared OpenFActScore + RECALL driver for the final experiments.
#
# This is the factscore/recall half of the pipeline. It always runs on Lisa
# (the AFV judge is Gemma-3-4B, small). For the Lisa-native models (OLMo, Qwen)
# it is the tail of the run; for the Homer models (Llama-70B, gpt-oss-120b) it
# is a SEPARATE step launched here against the chains produced on Homer.
#
# Two stages, both resumable:
#   1. forward OFS (precision): scripts/<model>/_common/openfactscore_eval.py
#        -> writes  <chains>_openfactscore.csv  +  <chains>_openfactscore_details.csv
#      The _details file holds the atomic E_0 facts that the recall step reuses.
#   2. recall: per-dataset script reusing those E_0 facts (no AFG re-extraction)
#        musique      -> scripts/600q/openfactscore_recall_600q.py        (Gemma AFV)
#        fictionalqa  -> scripts/fictionalqa/openfactscore_recall_fictionalqa.py (Gemma AFV)
#        newsqa       -> scripts/newsqa/recall_nli_eval_newsqa.py         (NLI cross-encoder)
#
# Forward OFS resumes per-row; recall resumes per (chain, step). Re-run after an
# interruption and it continues.
#
# Required variables (set by the caller):
#   MODEL_DIR    scripts/<model>   (olmo32b | qwen30b | llama70b | gptoss120b)
#   DATASET      musique | newsqa | fictionalqa
#   CHAINS       path to the rewriting_chains_*.csv produced by the forward run
# Optional:
#   AFV_USE_4BIT (default 1 -> pass --use-4bit to the Gemma AFV judge)
#   TOPIC_MODE   forward-OFS topic mode (default: qid)
#   GAMMA        recall length-penalty gamma (default: pipeline default = 10)
set -euo pipefail

run_ofs_recall() {
  : "${MODEL_DIR:?set MODEL_DIR}"
  : "${DATASET:?set DATASET}"
  : "${CHAINS:?set CHAINS}"

  if [ ! -f "$CHAINS" ]; then
    echo "[ERROR] chains CSV not found: $CHAINS" >&2
    echo "        Run the forward launcher first (it produces the chains)." >&2
    return 1
  fi

  local PY="scripts/${MODEL_DIR}"
  local TOPIC_MODE="${TOPIC_MODE:-qid}"
  local STEM="${CHAINS%.csv}"
  local DETAILS="${STEM}_openfactscore_details.csv"

  local AFV_ARGS=()
  [ "${AFV_USE_4BIT:-1}" = "1" ] && AFV_ARGS+=(--use-4bit)

  # The AFG/AFV judges (Gemma) trip TorchDynamo on Lisa; disable it (matches the
  # pipe311 OFS env note in the old Llama launcher).
  export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

  echo ""
  echo "============================================================"
  echo "  OPENFACTSCORE + RECALL — ${MODEL_DIR} / ${DATASET}"
  echo "============================================================"
  echo "  CHAINS   = $CHAINS"
  echo "  DETAILS  = $DETAILS"
  echo "  AFV      = Gemma-3-4B  (use_4bit=${AFV_USE_4BIT:-1})"
  echo "============================================================"

  # ---- 1/2 forward OFS (precision) — produces the E_0 facts details file ---- #
  echo ""
  echo "########## 1/2 — OpenFActScore forward (precision) ##########"
  # ${ARR[@]+"${ARR[@]}"} safely expands a possibly-empty array under `set -u`
  # (bash 3.2 errors on a bare "${ARR[@]}" when empty).
  python "$PY/_common/openfactscore_eval.py" \
    --input "$CHAINS" --topic-mode "$TOPIC_MODE" ${AFV_ARGS[@]+"${AFV_ARGS[@]}"}

  # ---- 2/2 recall — reuses E_0 facts from the details file above ------------ #
  echo ""
  echo "########## 2/2 — Recall (E_0 facts surviving in E_k) ##########"
  case "$DATASET" in
    musique)
      local GAMMA_ARGS=(); [ -n "${GAMMA:-}" ] && GAMMA_ARGS+=(--gamma "$GAMMA")
      python scripts/600q/openfactscore_recall_600q.py \
        --chain-csv "$CHAINS" --details-csv "$DETAILS" \
        ${AFV_ARGS[@]+"${AFV_ARGS[@]}"} ${GAMMA_ARGS[@]+"${GAMMA_ARGS[@]}"}
      ;;
    fictionalqa)
      local GAMMA_ARGS=(); [ -n "${GAMMA:-}" ] && GAMMA_ARGS+=(--gamma "$GAMMA")
      python scripts/fictionalqa/openfactscore_recall_fictionalqa.py \
        --chain-csv "$CHAINS" --details-csv "$DETAILS" \
        ${AFV_ARGS[@]+"${AFV_ARGS[@]}"} ${GAMMA_ARGS[@]+"${GAMMA_ARGS[@]}"}
      ;;
    newsqa)
      # NewsQA recall is NLI-based (cross-encoder), no Gemma AFV / gamma.
      python scripts/newsqa/recall_nli_eval_newsqa.py \
        --details "$DETAILS" --chains "$CHAINS" \
        --output "${STEM}_recall_nli.csv"
      ;;
    *) echo "[ERROR] unknown DATASET=$DATASET" >&2; return 1 ;;
  esac

  echo ""
  echo "############################################################"
  echo "  OFS + RECALL DONE — outputs next to the chains CSV:"
  echo "    ${STEM}_openfactscore.csv"
  echo "    ${STEM}_openfactscore_details.csv"
  if [ "$DATASET" = "newsqa" ]; then
    echo "    ${STEM}_recall_nli.csv"
  else
    echo "    ${STEM}_openfactscore_recall.csv"
    echo "    ${STEM}_openfactscore_recall_details.csv"
  fi
  echo "############################################################"
}
