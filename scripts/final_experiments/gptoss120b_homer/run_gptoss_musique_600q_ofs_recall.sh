#!/bin/bash
# OFS + RECALL for gpt-oss-120b · MuSiQue 600q — runs on LISA.
# AFV judge is Gemma-3-4B (small) → Lisa. Operates on the Homer chains.
#
# Prereq: run_gptoss_musique_600q_forward.sh produced
#   results/final/gptoss120b/musique_600q/rewriting_chains_musique_600q.csv
# (sync from Homer to Lisa under the same path if the NAS results dir isn't shared).
#
# Fully RESUMABLE. Usage:
#   tmux new -s gptoss_musique_ofs
#   bash scripts/final_experiments/gptoss120b_homer/run_gptoss_musique_600q_ofs_recall.sh \
#       2>&1 | tee logs/final_gptoss_musique_600q_ofs_recall.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env/env_lisa.sh"
source "$HERE/../_env/lib_ofs.sh"

export MODEL_DIR="gptoss120b"
export DATASET="musique"
export OUT_DIR="${OUT_DIR:-results/final/gptoss120b/musique_600q}"
export CHAINS="${CHAINS:-$OUT_DIR/rewriting_chains_musique_600q.csv}"
export TOPIC_MODE=qid
export AFV_USE_4BIT=1

run_ofs_recall

echo ""
echo "ALL DONE — gpt-oss MuSiQue 600q OFS+recall: $OUT_DIR"
