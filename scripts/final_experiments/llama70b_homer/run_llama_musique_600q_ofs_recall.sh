#!/bin/bash
# OFS + RECALL for Llama-3.1-70B · MuSiQue 600q — runs on LISA.
# Operates on the chains produced by the Homer forward run. The AFV judge is
# Gemma-3-4B (small), which is why this half runs on Lisa, not Homer.
#
# Prereq: run_llama_musique_600q_forward.sh has produced
#   results/final/llama70b/musique_600q/rewriting_chains_musique_600q.csv
# (copy/sync it from Homer to Lisa under the same path if the boxes don't share
#  the results dir on the NAS).
#
# Fully RESUMABLE. Usage:
#   tmux new -s llama_musique_ofs
#   bash scripts/final_experiments/llama70b_homer/run_llama_musique_600q_ofs_recall.sh \
#       2>&1 | tee logs/final_llama_musique_600q_ofs_recall.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_lisa.sh"
source "$HERE/../_env/lib_ofs.sh"

export MODEL_DIR="llama70b"
export DATASET="musique"
export OUT_DIR="${OUT_DIR:-results/final/llama70b/musique_600q}"
export CHAINS="${CHAINS:-$OUT_DIR/rewriting_chains_musique_600q.csv}"
export TOPIC_MODE=qid
export AFV_USE_4BIT=1

run_ofs_recall

echo ""
echo "ALL DONE — Llama MuSiQue 600q OFS+recall: $OUT_DIR"
