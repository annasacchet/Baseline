# OLMo-3.1-32B-Instruct rewriting pipeline (bf16)

Same rewriting + eval pipeline as `scripts/llama70b/`, retargeted to
`allenai/OLMo-3.1-32B-Instruct` served in **bf16** via vLLM with tensor
parallelism across two GPUs.

The rewriter, the Answer F1 QA model, and the perplexity model are all OLMo-
3.1-32B-Instruct, so PPL / F1 are computed under the rewriter's own
distribution.

Judges that were **not** the rewriter stay as-is for consistency with previous
runs:
- BERTScore  → `roberta-large` (layer 17)
- BLEURT     → `BLEURT-20`
- OFS-AFG    → `allenai/OLMo-2-1124-7B-SFT`
- OFS-AFV    → `google/gemma-3-4b-it`  (see `feedback_afv_model_consistency`)

## Layout

```
scripts/olmo32b/
├── _common/                  # shared modules + dataset-agnostic eval scripts
│   ├── olmo_constants.py     # model id, prompts, instructions
│   ├── olmo_model.py         # load_olmo() — HF transformers helper (bf16/4-bit)
│   ├── olmo_vllm.py          # load_vllm() + generate_batch_vllm()
│   ├── f1_utils.py           # SQuAD-style F1
│   ├── bertscore_eval.py     # --input <CSV>
│   ├── bleurt_eval.py        # --input <CSV> [--f1-csv <CSV>]
│   ├── perplexity_eval.py    # --input <CSV>
│   └── openfactscore_eval.py # --input <CSV> --topic-mode {qid,first-line,question}
├── musique/
│   ├── rewriting_pipeline_musique.py        # vLLM
│   ├── self_refine_pipeline_musique.py      # RQ3: Rewriter/Critic/Refiner
│   └── answer_f1_eval_musique.py            # vLLM
├── newsqa/
│   ├── rewriting_pipeline_newsqa.py         # vLLM
│   ├── self_refine_pipeline_newsqa.py       # RQ3
│   └── answer_f1_eval_newsqa.py             # vLLM
├── fictionalqa/
│   ├── rewriting_pipeline_fictionalqa.py    # vLLM
│   ├── self_refine_pipeline_fictionalqa.py  # RQ3
│   └── answer_f1_eval_fictionalqa.py        # vLLM
└── launch/
    ├── env_homer.sh                              # source'd by the others
    ├── launch_smoke_homer.sh                     # 1 q × 3 datasets, end-to-end
    ├── launch_smoke_full_musique_homer.sh        # 1 q, all 12 chains, full eval suite
    ├── launch_musique_homer.sh
    ├── launch_newsqa_homer.sh
    ├── launch_fictionalqa_homer.sh
    ├── launch_self_refine_musique_homer.sh       # RQ3: full self-refine pipeline
    ├── launch_self_refine_newsqa_homer.sh
    └── launch_self_refine_fictionalqa_homer.sh
```

Outputs go to `results/qwen30b/{musique,newsqa,fictionalqa}/`.

## Server requirements (Homer)

- ≥ 2 GPUs with ≥ 80 GB combined VRAM (e.g. 2× RTX A6000 48GB). In bf16 the
  OLMo-3.1-32B weights are ~64 GB; vLLM with TP=2 splits them across both GPUs
  and reserves the rest for KV cache.
- `OLMo-3.1-32B-Instruct` is open-weights — `HF_TOKEN` is **not** required
  (set one anyway to avoid HF rate limits).
- HF cache on NAS (`/mnt/dmif-nas/mitel/sacchet/hf_cache`) — see
  `feedback_lisa_hf_cache`. The Homer launcher exports it for you.
- conda env `baseline` with `vllm`, `transformers >= 4.45`, `bitsandbytes`
  (only used by the optional 4-bit OFS path), `accelerate`, `bert_score`,
  `evaluate`, `rank_bm25`, `nltk`, `bleurt` (`pip install git+https://github.com/google-research/bleurt`).

## Smoke test (≤ 30 min)

```bash
ssh sacchet@homer.dimi.uniud.it
tmux new -s olmo_smoke
bash ~/Baseline/scripts/olmo32b/launch/launch_smoke_homer.sh \
  2>&1 | tee ~/Baseline/logs/olmo_smoke.log
```

For the full MuSiQue smoke (1 question × 12 chains, 6 eval steps):

```bash
bash ~/Baseline/scripts/olmo32b/launch/launch_smoke_full_musique_homer.sh \
  2>&1 | tee ~/Baseline/logs/olmo_smoke_full_musique.log
```

## Full runs

```bash
bash ~/Baseline/scripts/olmo32b/launch/launch_musique_homer.sh
bash ~/Baseline/scripts/olmo32b/launch/launch_newsqa_homer.sh
bash ~/Baseline/scripts/olmo32b/launch/launch_fictionalqa_homer.sh
```

Knobs (env vars before launching):
- `N_PER_HOP=200` / `N_QUESTIONS=100` / `N_FICTIONS=100` — sample size
- `MAX_MODEL_LEN=8192` / `GPU_MEM_UTIL=0.90` — vLLM tuning

All scripts are **resumable**: re-running the same command picks up where the
previous one left off (per-chain or per-row deduplication on the output CSV).

## Self-Refine (RQ3)

Same Rewriter / Critic / Refiner loop as `scripts/15q/self_refine_pipeline.py`,
ported to OLMo-3.1-32B-Instruct via vLLM. All three roles use the same model;
generations are batched across the 12 chains of each question for each phase,
which keeps both GPUs busy under the 3× compute multiplier of self-refine.

E0 is sourced from the corresponding baseline `rewriting_chains_*.csv` when
present (byte-identical to RQ1); otherwise it is rebuilt from the dataset.
Output CSV adds `draft_text`, `critic_feedback`, `draft_n_tokens` next to the
standard chain columns — the existing eval scripts ignore the extras.

```bash
bash ~/Baseline/scripts/olmo32b/launch/launch_self_refine_musique_homer.sh
bash ~/Baseline/scripts/olmo32b/launch/launch_self_refine_newsqa_homer.sh
bash ~/Baseline/scripts/olmo32b/launch/launch_self_refine_fictionalqa_homer.sh
```
