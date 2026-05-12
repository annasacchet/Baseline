# NewsQA Pipeline Smoke Test Guide

## Quick Start (Lisa, 5 min setup)

### Prerequisites
- VPN connected (MITEL lab access)
- SSH key configured (see lab wiki)
- Dataset at `/mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv` on NAS

### Step 1: SSH to Lisa
```bash
ssh sacchet@lisa.dimi.uniud.it
```

### Step 2: Check GPU
```bash
nvidia-smi
# Should show 2× RTX 3090 (24GB each)
```

### Step 3: Start tmux session
```bash
tmux new -s newsqa
```

### Step 4: Run full pipeline smoke test
```bash
cd ~/Baseline
bash scripts/newsqa/run_full_pipeline_smoke.sh
```

**What this does:**
1. **Rewriting** (Stage 1) — loads 1 story, applies 4 instructions (style/formality, style/paraphrase, content/shorten, content/elaborate) × 3 wordings = 12 chains, 4 steps each (E0→E1→E2→E3) = 48 rows. Uses OLMo-3.1-32B-Instruct in **4-bit**.

2. **Answer F1** (Stage 2) — prompts OLMo-3.1 (4-bit) to answer the question on each chain step, computes F1 against gold answer + aliases using SQuAD normalization.

3. **OpenFActScore** (Stage 3) — runs AFG→AFV on all 12 chains (skipped if demons.json not available).

**Expected time:** ~15–20 min total on Lisa (4-bit quantization is slower than Homer bf16).

**Expected output:**
```
results/newsqa/rewriting_chains_newsqa_smoke.csv           (48 rows)
results/newsqa/rewriting_chains_newsqa_smoke_answer_f1.csv (48 rows + deduplicated E0)
results/newsqa/rewriting_chains_newsqa_smoke_openfactscore.csv (optional, ~12 rows)
```

---

## Alternative: Run individual stages

### Only rewriting (Stage 1)
```bash
python scripts/newsqa/rewriting_pipeline_newsqa.py \
  --model allenai/OLMo-3.1-32B-Instruct \
  --dataset /mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv \
  --output results/newsqa/rewriting_chains_newsqa_smoke.csv \
  --n-questions 1 \
  --n-iterations 3 \
  --smoke-test \
  --use-4bit
```

### Only Answer F1 (Stage 2, needs Stage 1 output)
```bash
python scripts/newsqa/answer_f1_eval_newsqa.py \
  --input results/newsqa/rewriting_chains_newsqa_smoke.csv \
  --output results/newsqa/rewriting_chains_newsqa_smoke_answer_f1.csv \
  --model allenai/OLMo-3.1-32B-Instruct \
  --batch-size 4 \
  --smoke-test \
  --use-4bit
```

### Only OpenFActScore (Stage 3, needs Stage 1 output)
```bash
python scripts/newsqa/openfactscore_eval_newsqa.py \
  --input results/newsqa/rewriting_chains_newsqa_smoke.csv \
  --limit 12 \
  --use-4bit
```

---

## Homer Alternative (bf16, no quantization)

If you prefer bf16 full precision on Homer (2× RTX A6000 48GB each, cleaner F1 numbers):

```bash
ssh sacchet@homer.dimi.uniud.it
cd ~/Baseline
tmux new -s newsqa
# Same command, but Homer has enough VRAM to skip --use-4bit
bash scripts/newsqa/run_full_pipeline_smoke.sh  # --use-4bit is optional here
```

**Difference:** Homer numbers will be slightly different from Lisa 4-bit (numerical precision). For official results, keep Lisa 4-bit consistent with your 300q MuSiQue runs.

---

## Troubleshooting

### "NEWSQA_DATASET not found"
Check that `/mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv` exists:
```bash
ls -lh /mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv
```

If missing, copy from local:
```bash
# On your Mac:
scp combined-newsqa-data-v1.csv sacchet@lisa.dimi.uniud.it:/mnt/dmif-nas/mitel/sacchet/
```

### "Out of memory" on Lisa
4-bit quantization should fit, but if OOM occurs:
- Reduce `--batch-size` in Answer F1 (default 4 → try 2)
- Reduce `--max-new-tokens` (default 4096 for rewriting, 96 for QA)

### "OLMo-3.1 not in HF cache"
First run will download (~65GB) to `/mnt/dmif-nas/mitel/sacchet/hf_cache/hub/`. This is cached for future runs.

### tmux detach/reattach
```bash
Ctrl+b d       # detach (leaves process running)
tmux attach -t newsqa    # reattach later
tmux kill-session -t newsqa  # kill session
```

---

## Understanding the output

### rewriting_chains_newsqa_smoke.csv
One row per (chain, step):
- `qid`: story ID (e.g., `./cnn/stories/abc123.story`)
- `question`: crowdsourced question
- `gold_answer`: primary validated answer span
- `gold_answer_aliases`: all validated spans with agreement ≥2 (||–joined)
- `group`: `style` or `content`
- `instruction_type`: `formality`, `paraphrase`, `shorten`, `elaborate`
- `run`: 0, 1, or 2 (three wordings per instruction)
- `step`: 0 (original text E_0), 1, 2, 3 (rewrites)
- `text`: the actual story/rewrite
- `n_tokens`: token count

### rewriting_chains_newsqa_smoke_answer_f1.csv
Same structure + QA results:
- `predicted_answer`: what OLMo-3.1 extracted for the question
- `matched_reference`: which gold alias had the highest F1
- `answer_f1`: F1 score (0–1)

### rewriting_chains_newsqa_smoke_openfactscore.csv
One row per (chain, step) with source-faithfulness metrics:
- `n_facts`: atomic facts extracted by AFG
- `n_supported`, `n_not_supported`: validation results from AFV
- `init_score`: n_supported / n_facts
- `factscore`: init_score × length penalty

---

## Next Steps (after smoke test)

Once smoke test passes, scale to full 300q (or your desired size):

```bash
# Full run: 300 stories
python scripts/newsqa/rewriting_pipeline_newsqa.py \
  --model allenai/OLMo-3.1-32B-Instruct \
  --dataset /mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv \
  --output results/newsqa/rewriting_chains_newsqa_300.csv \
  --n-questions 300 \
  --n-iterations 3 \
  --use-4bit
```

This will take ~4–6 hours on Lisa (12 chains × 4 steps × avg 30–40 seconds per gen).

For resumability (e.g., if process crashes), the pipeline auto-resumes from `results/newsqa/rewriting_chains_newsqa_300.csv` if it already exists.
