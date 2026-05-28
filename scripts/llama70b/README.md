# Llama-3.1-70B-Instruct rewriting pipeline (4-bit NF4)

Same rewriting + eval pipeline as `scripts/600q/`, `scripts/newsqa/`, and
`scripts/fictionalqa/`, but with `meta-llama/Llama-3.1-70B-Instruct` (4-bit NF4)
substituted everywhere OLMo-3.1-32B-Instruct was used (rewriter, Answer F1 QA
model, perplexity model).

Judges that were **not** OLMo-3.1 stay as-is for consistency with previous runs:
- BERTScore  → `roberta-large` (layer 17)
- BLEURT     → `BLEURT-20`
- OFS-AFG    → `allenai/OLMo-2-1124-7B-SFT`
- OFS-AFV    → `google/gemma-3-4b-it`  (see `feedback_afv_model_consistency`)

## Layout

```
scripts/llama70b/
├── _common/                  # shared modules + dataset-agnostic eval scripts
│   ├── llama_constants.py    # model id, prompts, instructions
│   ├── llama_model.py        # load_llama() — 4-bit NF4 helper
│   ├── f1_utils.py           # SQuAD-style F1
│   ├── bertscore_eval.py     # --input <CSV>
│   ├── bleurt_eval.py        # --input <CSV> [--f1-csv <CSV>]
│   ├── perplexity_eval.py    # --input <CSV>
│   └── openfactscore_eval.py # --input <CSV> --topic-mode {qid,first-line,question}
├── musique/
│   ├── rewriting_pipeline_musique.py
│   ├── self_refine_pipeline_musique.py   # RQ3: Rewriter/Critic/Refiner via vLLM
│   └── answer_f1_eval_musique.py
├── newsqa/
│   ├── rewriting_pipeline_newsqa.py
│   ├── self_refine_pipeline_newsqa.py    # RQ3 (vLLM AWQ-INT4)
│   └── answer_f1_eval_newsqa.py
├── fictionalqa/
│   ├── rewriting_pipeline_fictionalqa.py
│   ├── self_refine_pipeline_fictionalqa.py # RQ3 (vLLM AWQ-INT4)
│   └── answer_f1_eval_fictionalqa.py
└── launch/
    ├── env_homer.sh                              # source'd by the others
    ├── launch_smoke_homer.sh                     # 1 q × 3 datasets, end-to-end
    ├── launch_musique_homer.sh
    ├── launch_newsqa_homer.sh
    ├── launch_fictionalqa_homer.sh
    ├── launch_self_refine_musique_homer.sh       # RQ3: full self-refine pipeline
    ├── launch_self_refine_newsqa_homer.sh
    └── launch_self_refine_fictionalqa_homer.sh
```

Outputs go to `results/llama70b/{musique,newsqa,fictionalqa}/`.

## Server requirements (Homer)

- ≥ 2 GPUs with ≥ 40 GB total VRAM (e.g. 2× RTX A6000 48GB). At 4-bit NF4 the
  Llama-3.1-70B weights occupy ~40 GB.
- `HF_TOKEN` exported (Llama-3.1 is gated).
- HF cache on NAS (`/mnt/dmif-nas/mitel/sacchet/hf_cache`) — see
  `feedback_lisa_hf_cache`. The Homer launcher exports it for you.
- conda env `baseline` with `transformers >= 4.45`, `bitsandbytes`, `accelerate`,
  `bert_score`, `evaluate`, `rank_bm25`, `nltk`, `bleurt` (`pip install git+https://github.com/google-research/bleurt`).

## Smoke test (≤ 30 min)

```bash
ssh sacchet@homer.dimi.uniud.it
tmux new -s llama_smoke
bash ~/Baseline/scripts/llama70b/launch/launch_smoke_homer.sh \
  2>&1 | tee ~/Baseline/logs/llama_smoke.log
```

## Full runs

```bash
bash ~/Baseline/scripts/llama70b/launch/launch_musique_homer.sh
bash ~/Baseline/scripts/llama70b/launch/launch_newsqa_homer.sh
bash ~/Baseline/scripts/llama70b/launch/launch_fictionalqa_homer.sh
```

Knobs (env vars before launching):
- `N_PER_HOP=200` / `N_QUESTIONS=100` / `N_FICTIONS=100` — sample size
- `BATCH=4` — chains advanced in parallel per `generate()` call. Lower to 2
  if you OOM on long contexts (NewsQA articles).

All scripts are **resumable**: re-running the same command picks up where the
previous one left off (per-chain or per-row deduplication on the output CSV).

## Self-Refine (RQ3)

Same Rewriter / Critic / Refiner loop as `scripts/15q/self_refine_pipeline.py`,
ported to Llama-3.1-70B-AWQ via vLLM. All three roles use the same model;
generations are batched across the 12 chains of each question (`12 prompts ×
3 phases × N iterations` per question), which is the only way to keep both
GPUs busy under the 3× compute multiplier of self-refine.

E0 is sourced from the corresponding baseline `rewriting_chains_*.csv` when
present (byte-identical to RQ1); otherwise it is rebuilt from the dataset.
Output CSV adds `draft_text`, `critic_feedback`, `draft_n_tokens` next to the
standard chain columns — the existing eval scripts ignore the extras.

```bash
bash ~/Baseline/scripts/llama70b/launch/launch_self_refine_musique_homer.sh
bash ~/Baseline/scripts/llama70b/launch/launch_self_refine_newsqa_homer.sh
bash ~/Baseline/scripts/llama70b/launch/launch_self_refine_fictionalqa_homer.sh
```
