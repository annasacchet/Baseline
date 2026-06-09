# Final experiments — 4 models × 3 datasets × 600 questions

This folder holds the launch scripts for the final experiment matrix. Each
experiment does **iterative rewriting (3 steps)** and then computes every metric:
BERTScore (baseline + consecutive), BLEURT, Answer F1, FactScore (OpenFActScore),
Recall, and token length.

All launchers are **resumable**: re-run the exact same command after any
interruption and every step continues from where it stopped (rewriting skips
chains already in the CSV; Answer-F1 / perplexity / OFS / recall skip rows
already scored; BERTScore / BLEURT recompute, which is cheap).

## The matrix

| Model | Server | Datasets | FactScore + Recall |
|-------|--------|----------|--------------------|
| OLMo-3.1-32B-Instruct      | **Lisa**  | musique, newsqa, fictionalqa | run locally (in the same launcher) |
| Qwen3-30B-A3B-Instruct-2507| **Lisa**  | musique, newsqa, fictionalqa | run locally (in the same launcher) |
| Llama-3.1-70B-Instruct     | **Homer** | musique, newsqa, fictionalqa | **separate** script, run on Lisa |
| openai/gpt-oss-120b        | **Homer** | musique, newsqa, fictionalqa | **separate** script, run on Lisa |

Sampling per experiment = **600 questions**:
- **MuSiQue**: `--n-per-hop 200` → 200 per hop class (2-hop / 3-hop / 4-hop) = **600**, balanced.
- **NewsQA**: `--n-questions 600`.
- **FictionalQA**: `--n-fictions 600` (one question per fiction document, pulled from the HF Hub).

## Why the Homer models are split into two scripts

For the two large Homer models (Llama-70B, gpt-oss-120b) the request is: run
**everything except FactScore and Recall** on Homer, and run FactScore + Recall
**separately**. FactScore/Recall use the small **Gemma-3-4B** AFV judge, which
lives on **Lisa** — so each Homer model gets:

- `run_<model>_<dataset>_600q_forward.sh` — runs on **Homer** (rewriting, Answer
  F1, BERTScore, BLEURT, perplexity, token length).
- `run_<model>_<dataset>_600q_ofs_recall.sh` — runs on **Lisa**, against the
  chains produced by the forward run.

If Homer and Lisa do not share the `results/` dir on the NAS, copy the
`rewriting_chains_*.csv` to Lisa under the same path before running the
`*_ofs_recall.sh` script (or override `CHAINS=...`).

The Lisa-native models (OLMo, Qwen) run the whole thing — forward + OFS + recall
— in a single launcher.

## How to run

Always run inside `tmux` (jobs take many hours). Example:

```bash
# On Lisa
tmux new -s olmo_musique
bash scripts/final_experiments/olmo32b_lisa/run_olmo_musique_600q.sh \
    2>&1 | tee logs/final_olmo_musique_600q.log

# On Homer (forward), then on Lisa (OFS + recall)
tmux new -s llama_musique
bash scripts/final_experiments/llama70b_homer/run_llama_musique_600q_forward.sh \
    2>&1 | tee logs/final_llama_musique_600q_forward.log
# ... later, on Lisa:
bash scripts/final_experiments/llama70b_homer/run_llama_musique_600q_ofs_recall.sh \
    2>&1 | tee logs/final_llama_musique_600q_ofs_recall.log
```

Set `HF_TOKEN` first (Llama-70B and the Gemma AFV judge are gated):
`export HF_TOKEN=hf_...`

## Layout

```
final_experiments/
├── _env/
│   ├── env_lisa.sh      # shared env for Lisa (HF cache on NAS, dataset paths)
│   ├── env_homer.sh     # shared env for Homer
│   ├── lib_forward.sh   # run_forward(): rewriting→F1→BERTScore→BLEURT→PPL
│   └── lib_ofs.sh       # run_ofs_recall(): forward OFS (precision) → recall
├── olmo32b_lisa/        # 3 launchers (full pipeline, OFS+recall local)
├── qwen30b_lisa/        # 3 launchers (full pipeline, OFS+recall local)
├── llama70b_homer/      # 3 forward + 3 ofs_recall launchers
└── gptoss120b_homer/    # 3 forward + 3 ofs_recall launchers
```

The launchers are thin: they `source` the env + the libraries, set a handful of
variables (model dir, dataset, sample size, backend, context length), and call
`run_forward` and/or `run_ofs_recall`.

## Backends per model

The underlying per-model pipelines live in `scripts/<model>/` (cloned from the
validated `scripts/olmo32b/` tree; only the model ID in
`_common/olmo_constants.py` and the package path differ). Backends:

- **OLMo-3.1-32B (Lisa)**: HF backend (`--backend hf`) + 4-bit (bitsandbytes).
  OLMo-3.1 has **no native vLLM kernel**, so vLLM would just fall back to its
  slower Transformers path — we use HF directly for MuSiQue. The NewsQA /
  FictionalQA rewriting pipelines are vLLM-only (they ignore `--backend`) and
  run on vLLM's Transformers fallback.
- **Qwen3-30B-A3B (Lisa)**: native vLLM (Qwen3 has a kernel), bf16. Set
  `QUANT=bitsandbytes` if VRAM is tight.
- **Llama-3.1-70B (Homer)**: MuSiQue uses the AWQ-INT4 checkpoint via vLLM
  (TP=2). NewsQA / FictionalQA rewriting + Answer-F1 use HF transformers 4-bit.
- **gpt-oss-120b (Homer)**: native vLLM, TP=2.

The launchers don't hardcode which flags each pipeline accepts: `lib_forward.sh`
introspects every pipeline's `--help` and passes each generation flag
(`--backend`, `--max-model-len`, `--quantization`, `--batch-size`, `--resume`,
…) **only if that specific script supports it**. This absorbs the differences
between the vLLM and HF/bnb pipelines without per-script special-casing.

## Tuning

Every launcher reads overridable env vars (with sensible defaults baked in):

```bash
# smaller smoke run
N_PER_HOP=5 bash scripts/final_experiments/olmo32b_lisa/run_olmo_musique_600q.sh   # musique
N_ITEMS=20  bash scripts/final_experiments/qwen30b_lisa/run_qwen_newsqa_600q.sh    # newsqa/fictionalqa

# other knobs: N_ITERATIONS, TEMPERATURE, SEED, MAX_NEW_TOKENS, MAX_MODEL_LEN,
#              GPU_MEM_UTIL, TP, BACKEND, QUANT, SKIP_PPL=1, OUT_DIR
```

## Outputs

Everything lands under `results/final/<model>/<dataset>_600q/`:

```
rewriting_chains_<dataset>_600q.csv                      # chains + n_tokens (token length)
rewriting_chains_<dataset>_600q_answer_f1.csv
rewriting_chains_<dataset>_600q_bertscore.csv            # Baseline + Consecutive
rewriting_chains_<dataset>_600q_bleurt.csv               # baseline + consecutive + answer
rewriting_chains_<dataset>_600q_perplexity.csv
rewriting_chains_<dataset>_600q_openfactscore.csv        # FactScore (precision)
rewriting_chains_<dataset>_600q_openfactscore_details.csv# E_0 atomic facts (reused by recall)
rewriting_chains_<dataset>_600q_openfactscore_recall.csv # recall (musique / fictionalqa, Gemma AFV)
rewriting_chains_<dataset>_600q_recall_nli.csv           # recall (newsqa, NLI cross-encoder)
```
