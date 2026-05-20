#!/bin/bash
# Quick launcher for NewsQA smoke test on Lisa
# Just run: bash RUN_NEWSQA_NOW.sh

echo "🚀 Launching NewsQA smoke test on Lisa..."
echo ""
echo "This will:"
echo "  1. SSH to Lisa"
echo "  2. Create a new tmux session"
echo "  3. Run the full pipeline (rewriting → Answer F1 → OpenFActScore)"
echo ""
echo "Time: ~15–20 minutes"
echo ""

read -p "Press Enter to continue, Ctrl+C to cancel..."

ssh sacchet@lisa.dimi.uniud.it << 'REMOTE'
  # On remote (Lisa):
  tmux new-session -d -s newsqa
  tmux send-keys -t newsqa "cd ~/Baseline && bash scripts/newsqa/run_full_pipeline_smoke.sh" Enter
  
  echo ""
  echo "✓ tmux session 'newsqa' created and running on Lisa"
  echo ""
  echo "To monitor progress, reattach with:"
  echo "  tmux attach -t newsqa"
  echo ""
REMOTE

echo "Done! Check Lisa by running: tmux attach -t newsqa"
