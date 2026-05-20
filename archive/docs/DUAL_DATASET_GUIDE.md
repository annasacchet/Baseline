# NewsQA + FictionalQA Parallel Pipeline Guide

## Quick Reference

### Smoke Test (1 story each, ~30 min total)

**On H100 96GB:**
```bash
bash run_both_parallel_h100.sh 1 1
```

**On Lisa separately:**
```bash
# Terminal 1: NewsQA
ssh sacchet@lisa.dimi.uniud.it
tmux new -s newsqa
cd ~/Baseline && bash scripts/newsqa/run_full_pipeline_smoke.sh

# Terminal 2: FictionalQA (same server, different GPU)
ssh sacchet@lisa.dimi.uniud.it
tmux new -s fictionalqa
cd ~/Baseline && bash scripts/fictionalqa/launch_lisa.sh
```

### Full Run (300 each, ~4.5 hours on H100)

```bash
bash run_both_parallel_h100.sh 300 300
```

---

## Detailed Setup

### Option 1: H100 Rental (Recommended for Full Run)

**Advantages:**
- Both datasets run in **parallel** (GPU 0 + GPU 1)
- 4-bit quantization fits comfortably (40–46 GB VRAM)
- **~4.5 hours total vs. 8 hours sequential**
- Cost: ~$6.75 for 300q+300f full run

**Setup on H100:**

```bash
# 1. Upload repo to H100 (or clone)
rsync -av ~/Desktop/Baseline/ h100_user@h100_ip:~/Baseline/

# 2. SSH in
ssh h100_user@h100_ip
cd ~/Baseline

# 3. Create conda env (if needed)
conda create -n baseline python=3.11 pytorch::pytorch pytorch::pytorch-cuda=12.1 -y
conda activate baseline
pip install -r requirements.txt

# 4. Set up HF cache (if /tmp has space)
export HF_HOME=/tmp/hf_cache
mkdir -p $HF_HOME

# 5. Add dataset path if local
export NEWSQA_DATASET=/path/to/combined-newsqa-data-v1.csv
# (If not local, script will download from HF)

# 6. Run parallel pipeline
bash run_both_parallel_h100.sh 300 300
```

**Monitor progress:**
```bash
# In another terminal
watch -n 5 nvidia-smi
```

---

### Option 2: Lisa Lab Servers (Sequential, Free)

**Advantages:**
- Free (lab GPU)
- No setup needed (repo already there)

**Disadvantages:**
- Sequential execution (NewsQA then FictionalQA)
- ~8 hours total vs. 4.5 on H100
- May need to wait for GPU availability

**Setup on Lisa:**

**Terminal 1 (NewsQA):**
```bash
ssh sacchet@lisa.dimi.uniud.it
tmux new -s newsqa
cd ~/Baseline
bash scripts/newsqa/run_full_pipeline_smoke.sh
# Or full run:
python scripts/newsqa/rewriting_pipeline_newsqa.py \
  --model allenai/OLMo-3.1-32B-Instruct \
  --dataset /mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv \
  --output results/newsqa/rewriting_chains_newsqa_300.csv \
  --n-questions 300 \
  --use-4bit
```

**Terminal 2 (FictionalQA, same server or wait):**
```bash
ssh sacchet@lisa.dimi.uniud.it
tmux new -s fictionalqa
cd ~/Baseline
bash scripts/fictionalqa/launch_lisa.sh
# Or full run:
python scripts/fictionalqa/rewriting_pipeline_fictionalqa.py \
  --model allenai/OLMo-3.1-32B-Instruct \
  --output results/fictionalqa/rewriting_chains_fictionalqa_300.csv \
  --n-fictions 300 \
  --use-4bit
```

---

## Model Stack (OLMo-3.1, 4-bit)

Both datasets use **identical models** for fair comparison:

| Task | Model | Precision | VRAM |
|------|-------|-----------|------|
| Rewriting | allenai/OLMo-3.1-32B-Instruct | 4-bit NF4 | ~18 GB |
| QA (Answer F1) | allenai/OLMo-3.1-32B-Instruct | 4-bit NF4 | ~16 GB |
| AFG (Atomic Facts) | allenai/OLMo-2-1124-7B-SFT | 4-bit NF4 | ~5 GB |
| AFV (Fact Validation) | google/gemma-3-4b-it | 4-bit NF4 | ~3 GB |

**Total simultaneous:** ~40–46 GB (fits in H100 single GPU, or H100 multi-GPU with 48GB each)

---

## Timing Breakdown

### Smoke Test (1 story + 1 fiction)

| Stage | NewsQA | FictionalQA | Parallel | Total |
|-------|--------|-------------|----------|-------|
| Rewriting (1 × 12 chains × 4 steps) | 5 min | 3 min | ~5 min | |
| Answer F1 | 2 min | 1 min | ~2 min | |
| OpenFActScore | 3 min | 2 min | ~3 min | |
| **TOTAL** | 10 min | 6 min | — | **~10 min** |

### Full Run (300 stories + 300 fictions)

| Stage | NewsQA | FictionalQA | Parallel | Note |
|-------|--------|-------------|----------|------|
| Rewriting | 2.5h | 1.5h | ~2.5h | NewsQA larger, 300 chains × 12 instructions |
| Answer F1 | 45 min | 25 min | ~45 min | Batch size 12, 3.6k texts per dataset |
| OpenFActScore | 1.5h | 1h | ~1.5h | AFG/AFV bottleneck |
| **TOTAL** | ~4.5h | ~3.5h | — | **~4.5h (parallel)** vs. 8h sequential |

---

## Output Files

After running, check `results/`:

```
results/
├── newsqa/
│   ├── rewriting_chains_newsqa_300.csv                  (12 chains × 4 steps × 300 stories)
│   ├── rewriting_chains_newsqa_300_answer_f1.csv        (F1 scores per text)
│   └── rewriting_chains_newsqa_300_openfactscore.csv    (source faithfulness)
│
└── fictionalqa/
    ├── rewriting_chains_fictionalqa_300.csv             (12 chains × 4 steps × 300 fictions)
    ├── rewriting_chains_fictionalqa_300_answer_f1.csv   (F1 scores per text)
    └── rewriting_chains_fictionalqa_300_openfactscore.csv (source faithfulness)
```

---

## Comparison & Analysis

Once both complete, you can analyze:

1. **Answer F1 by instruction type:**
   - NewsQA: real-world news articles
   - FictionalQA: fictional stories (tests pure information preservation)

2. **OpenFActScore differences:**
   - NewsQA: AFV might use prior knowledge (articles are real)
   - FictionalQA: AFV must rely purely on source text (facts are invented)

3. **Rewriting robustness:**
   - Which instructions cause more information loss?
   - How do the two datasets compare in stability?

---

## Troubleshooting

### H100: "CUDA out of memory"
Reduce batch size in Answer F1:
```bash
--batch-size 8  # instead of 12
```

### Lisa: GPU occupied
Use `CUDA_VISIBLE_DEVICES` to force a specific GPU:
```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/fictionalqa/launch_lisa.sh
```

### HF models not cached
First run downloads ~65 GB OLMo-3.1 to `$HF_HOME/hub/`. This takes 10–30 min depending on connection. Subsequent runs use cache.

### FictionalQA: "Dataset download failed"
The script automatically downloads from HF Hub. If that fails:
```bash
# Check internet connection
ping huggingface.co

# Or download manually:
huggingface-cli download jwkirchenbauer/fictionalqa --repo-type dataset
```

---

## Cost Summary

| Scenario | GPU | Time | Cost |
|----------|-----|------|------|
| Smoke test (H100) | H100 96GB | ~10 min | **$0.25** |
| Full run (H100) | H100 96GB | ~4.5h | **$6.75** |
| Full run (Lisa, free) | 2× RTX 3090 | ~8h | **Free** (wait time) |

**Recommendation:** If you have time, use Lisa (free). If you need results in hours, H100 rental is worth it.

---

## Next Steps

1. **Decide:** H100 rental vs. Lisa wait time?
2. **Run smoke test** first (10–30 min, validates setup)
3. **Check outputs:** Are metrics sensible?
4. **Run full pipeline** (300 each)
5. **Compare:** NewsQA vs. FictionalQA on rewriting robustness
