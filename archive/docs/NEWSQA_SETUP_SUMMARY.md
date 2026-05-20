# NewsQA Pipeline Setup — Complete Summary

## What Was Done

### 1. **Fixed Dataset Path**
- Changed hardcoded `/Users/annasacchet/combined-newsqa-data-v1.csv` → environment-based
- Now looks for `NEWSQA_DATASET` env var, falls back to `/mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv`
- File: `scripts/newsqa/rewriting_pipeline_newsqa.py` line 49–52

### 2. **Created Launch Scripts**
Three shell scripts in `scripts/newsqa/`:

**a) `run_full_pipeline_smoke.sh` (MAIN ORCHESTRATOR)**
- Runs complete pipeline: rewriting → Answer F1 → OpenFActScore
- Handles env var setup (HF_HOME, HF_HUB_CACHE, TRANSFORMERS_CACHE)
- Automatic 4-bit vs bf16 selection (configurable via `USE_4BIT` env var)
- Defaults to Lisa setup (4-bit) but works on Homer too

**b) `launch_lisa.sh`** — Lisa-specific smoke test launcher (4-bit, 2× RTX 3090)

**c) `launch_homer.sh`** — Homer-specific smoke test launcher (bf16, 2× RTX A6000)

### 3. **Created Documentation**
- **NEWSQA_SMOKE_TEST_GUIDE.md** — Detailed step-by-step guide (prerequisites, setup, troubleshooting, output schema)
- **QUICK_START.txt** — TL;DR reference card
- **NEWSQA_SETUP_SUMMARY.md** (this file) — What was done and why

### 4. **Updated Memory**
- Added `project_newsqa_pipeline.md` to memory system for future reference
- Updated `MEMORY.md` index

---

## What to Run Next

### Smoke Test (Recommended First)
```bash
ssh sacchet@lisa.dimi.uniud.it
tmux new -s newsqa
cd ~/Baseline
bash scripts/newsqa/run_full_pipeline_smoke.sh
```

**Expected:**
- ~15–20 minutes total
- Processes 1 story (12 chains × 4 steps = 48 rows)
- Outputs: 3 CSV files in `results/newsqa/`

### Full Run (300 stories)
Same command but without smoke-test flags:
```bash
python scripts/newsqa/rewriting_pipeline_newsqa.py \
  --model allenai/OLMo-3.1-32B-Instruct \
  --dataset /mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv \
  --output results/newsqa/rewriting_chains_newsqa_300.csv \
  --n-questions 300 \
  --n-iterations 3 \
  --use-4bit
```

**Expected:**
- ~4–6 hours on Lisa (12 instructions × 3 wordings × 300 stories = 10.8k chains, ~4 min per chain in 4-bit)

---

## Configuration

### Model Stack
- **Rewriter:** allenai/OLMo-3.1-32B-Instruct (you requested this)
- **QA model:** allenai/OLMo-3.1-32B-Instruct
- **AFG:** allenai/OLMo-2-1124-7B-SFT (atomic fact generation)
- **AFV:** google/gemma-3-4b-it (atomic fact validation)

### Dataset
- **File:** `/mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv`
- **Size:** 446 MB
- **Format:** CSV with columns: story_id, question, answer_char_ranges, validated_answers, story_text
- **"v1"** = version 1 (consolidated format)
- Pipeline selects 1 best question per story (by agreement ≥2 crowdworkers)

### Key Parameters
| Parameter | Rewriting | QA | AFG | AFV |
|-----------|-----------|-----|-----|-----|
| Max tokens | 4096 | 96 | 256 | 8 |
| Temperature | 0.7 | N/A (greedy) | N/A | N/A |
| Batch size | 1 | 4 | 1 | 1 |
| Precision | 4-bit (Lisa) | 4-bit (Lisa) | 4-bit | 4-bit |
| Precision | bf16 (Homer) | bf16 (Homer) | bf16 | bf16 |

---

## Understanding the Pipeline

### Stage 1: Rewriting (E₀ → E₁ → E₂ → E₃)
- Takes raw CNN article (E₀)
- Applies one instruction (e.g., "Make more formal") 3 times iteratively
- Instruction pool: `{style, content} × {formality, paraphrase, shorten, elaborate} × {3 wordings each}`
- Total per story: 4 groups × 3 instruction_types × 3 wordings = 36 chains
- Wait, that's wrong... let me recalculate:
  - Groups: 2 (style, content)
  - Instruction types per group: 2 (style={formality, paraphrase}, content={shorten, elaborate})
  - Wordings: 3 per instruction_type
  - **Total: 2 × 2 × 3 = 12 chains per story** ✓
- Each chain: E₀ + 3 iterative rewrites = 4 steps, 4 rows per chain
- **Total rows per story: 12 chains × 4 steps = 48 rows**

### Stage 2: Answer F1
- QA model answers the question on each chain text
- Compares predicted answer to gold answer + aliases using SQuAD-style normalization
- Deduplicates E₀ (since it's identical across all instruction_types of the same run)
- Broadcasts E₀ prediction back to all instruction_types

### Stage 3: OpenFActScore
- AFG breaks down each text into atomic facts (one statement per fact)
- AFV validates each fact against the **original article (E₀)** as source
- Computes: init_score = n_supported / n_facts, then applies length penalty
- This measures **content drift from original** (how much rewriting changed factual content)

---

## Why This Setup

### OLMo-3.1-32B-Instruct (not OLMo-2)
- You specifically requested OLMo-3.1 for rewriting
- It's already cached on Lisa (`/mnt/dmif-nas/mitel/sacchet/hf_cache/hub/`)
- Fits in 4-bit on Lisa (48GB total VRAM), can also run bf16 on Homer

### Lisa by default
- For consistency with your 300q MuSiQue setup (which runs OFS on Lisa in 4-bit)
- Homer is reserved for heavier runs, but works fine too

### 4-bit quantization on Lisa
- 32B models in full precision = ~65GB weights; Lisa has 48GB VRAM split across 2 GPUs
- 4-bit reduces to ~16GB, comfortable fit with `device_map="auto"`
- Slight numerical differences from bf16, but acceptable for evaluation metrics

---

## Files Changed / Created

```
Desktop/Baseline/
├── scripts/newsqa/
│   ├── rewriting_pipeline_newsqa.py    [MODIFIED] dataset path (line 49–52)
│   ├── run_full_pipeline_smoke.sh      [CREATED] main orchestrator
│   ├── launch_lisa.sh                  [CREATED] Lisa launcher
│   ├── launch_homer.sh                 [CREATED] Homer launcher
│   └── QUICK_START.txt                 [CREATED] TL;DR reference
├── NEWSQA_SMOKE_TEST_GUIDE.md          [CREATED] detailed guide
└── NEWSQA_SETUP_SUMMARY.md             [CREATED] this file

.claude/projects/.../memory/
├── project_newsqa_pipeline.md          [CREATED] pipeline config for future sessions
└── MEMORY.md                           [UPDATED] added entry for NewsQA
```

---

## Troubleshooting

See **NEWSQA_SMOKE_TEST_GUIDE.md** for detailed troubleshooting (OOM, missing dataset, env issues, etc).

Quick checklist:
- [ ] SSH key configured (lab wiki)
- [ ] VPN connected
- [ ] Can reach `/mnt/dmif-nas/mitel/sacchet/combined-newsqa-data-v1.csv`
- [ ] Can reach `/mnt/dmif-nas/mitel/sacchet/hf_cache/`
- [ ] Lisa has 2× RTX 3090 visible (`nvidia-smi`)
- [ ] `conda activate baseline` works
- [ ] Bash scripts are executable (`chmod +x`)

---

## Next Steps

1. **Run smoke test** on Lisa (15–20 min, validates entire pipeline)
2. **Check outputs** (3 CSV files with metrics)
3. **Scale to 300 stories** (4–6 hours)
4. **Compare results** with your ChatGPT baseline (if you have one)

Good luck! Questions? Check the guides or memory system.
