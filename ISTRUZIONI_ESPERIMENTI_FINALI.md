# Istruzioni — Esperimenti finali (4 modelli × 3 dataset × 600 domande)

Documento operativo: **comandi pronti da copia-incollare**. Ogni esperimento fa
rewriting iterativo (3 step) e calcola: BERTScore (baseline + consecutive),
BLEURT, Answer F1, FactScore (OpenFActScore), Recall, token length.

Tutti gli script sono **resume**: se un job si interrompe, rilancia *lo stesso
identico comando* e riparte da dove era rimasto.

---

## 0. Setup iniziale (una volta per sessione SSH, su OGNI server)

```bash
# Connettiti (VPN attiva) e vai nel repo
ssh sacchet@homer        # oppure: ssh sacchet@lisa.dimi.uniud.it
cd ~/Baseline

# Token HF (Llama-70B e il giudice Gemma-3-4B sono gated)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> Gli script sorgiano da soli env + cache HF su NAS + dataset path. **NON usano
> conda**: devi attivare TU il tuo virtualenv prima del comando (è quello che ha
> vLLM / transformers / le dipendenze):
>
> ```bash
> source ~/Baseline/.venv/bin/activate   # adatta il path del tuo venv
> ```
>
> Se nel PATH c'è solo `python3` (non `python`), gli `env_*.sh` creano da soli
> uno shim `python`→`python3`, quindi i pipeline girano comunque. Ma le
> dipendenze devono stare nel python attivo: **attiva il venv prima**.

### Come funziona tmux (job lunghi, ore)

Ogni lancio va dentro una sessione `tmux` dedicata, così il job continua anche
se cade la connessione SSH. Il pattern è sempre lo stesso:

```bash
tmux new -s NOME_SESSIONE                  # 1) crea/entra nella sessione
#   --- ora sei DENTRO tmux ---
cd ~/Baseline                              # 2) la sessione parte dalla home
source .venv/bin/activate                  # 3) attiva il venv (dipendenze)
export HF_TOKEN=hf_xxxx                     # 4) re-esporta il token DENTRO tmux
bash scripts/.../run_xxx.sh 2>&1 | tee logs/xxx.log   # 5) lancia
#   stacca senza fermare il job:   premi  Ctrl-b  poi  d
```

> ⚠️ Le variabili (`HF_TOKEN`, `CUDA_VISIBLE_DEVICES`, ecc.) vanno esportate
> **dentro** la sessione tmux: una nuova sessione tmux NON eredita gli export
> fatti prima nella shell SSH.

Gestione sessioni:

```bash
tmux ls                     # elenca le sessioni attive
tmux attach -t NOME         # rientra in una sessione
tmux kill-session -t NOME   # termina una sessione
```

### Scelta della GPU (Lisa)

Controlla quale GPU è libera e selezionala **dentro** tmux, prima del comando:

```bash
nvidia-smi                       # guarda Memory-Usage / GPU-Util
export CUDA_VISIBLE_DEVICES=1    # usa la GPU 1 (i launcher Lisa hanno già TP=1)
```

> Con una sola GPU lascia il default `TP=1` (già impostato). Dopo
> `CUDA_VISIBLE_DEVICES=1` quella GPU appare come `cuda:0` nei log: è normale.

Nei blocchi qui sotto il pattern è già scritto per esteso: copia l'intero blocco
di un esperimento, poi (dentro tmux) lancia la riga `bash ...`.

---

## 1. MODELLI SU LISA — OLMo-3.1-32B e Qwen3-30B (pipeline completa)

Su Lisa ogni launcher fa **tutto**: forward (rewriting → F1 → BERTScore → BLEURT
→ perplexity) **e** OFS + recall in coda. Un solo comando per esperimento.

> Ogni blocco è autonomo: crea la sessione tmux, poi DENTRO esporta token/GPU e
> lancia. Stacca con `Ctrl-b` `d`.

### OLMo-3.1-32B-Instruct

```bash
# === OLMo · MuSiQue — 600 domande (200 per hop: 2/3/4-hop) ===
tmux new -s olmo_musique
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx; export CUDA_VISIBLE_DEVICES=1
bash scripts/final_experiments/olmo32b_lisa/run_olmo_musique_600q.sh \
    2>&1 | tee logs/final_olmo_musique_600q.log
```

```bash
# === OLMo · NewsQA — 600 domande ===
tmux new -s olmo_newsqa
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx; export CUDA_VISIBLE_DEVICES=1
bash scripts/final_experiments/olmo32b_lisa/run_olmo_newsqa_600q.sh \
    2>&1 | tee logs/final_olmo_newsqa_600q.log
```

```bash
# === OLMo · FictionalQA — 600 domande ===
tmux new -s olmo_fictionalqa
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx; export CUDA_VISIBLE_DEVICES=1
bash scripts/final_experiments/olmo32b_lisa/run_olmo_fictionalqa_600q.sh \
    2>&1 | tee logs/final_olmo_fictionalqa_600q.log
```

### Qwen3-30B-A3B-Instruct-2507

```bash
# === Qwen · MuSiQue — 600 domande (200 per hop) ===
tmux new -s qwen_musique
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx; export CUDA_VISIBLE_DEVICES=1
bash scripts/final_experiments/qwen30b_lisa/run_qwen_musique_600q.sh \
    2>&1 | tee logs/final_qwen_musique_600q.log
```

```bash
# === Qwen · NewsQA — 600 domande ===
tmux new -s qwen_newsqa
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx; export CUDA_VISIBLE_DEVICES=1
bash scripts/final_experiments/qwen30b_lisa/run_qwen_newsqa_600q.sh \
    2>&1 | tee logs/final_qwen_newsqa_600q.log
```

```bash
# === Qwen · FictionalQA — 600 domande ===
tmux new -s qwen_fictionalqa
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx; export CUDA_VISIBLE_DEVICES=1
bash scripts/final_experiments/qwen30b_lisa/run_qwen_fictionalqa_600q.sh \
    2>&1 | tee logs/final_qwen_fictionalqa_600q.log
```

---

## 2. MODELLI SU HOMER — Llama-3.1-70B e gpt-oss-120b (in DUE fasi)

Per i modelli grandi: la parte **forward** (tutto tranne FactScore + Recall)
gira su **Homer**; FactScore + Recall girano **a parte su Lisa** (il giudice
Gemma-3-4B è piccolo e sta su Lisa).

### Fase A — FORWARD su HOMER

> Su Homer i modelli grandi usano **2 GPU** (TP=2): NON impostare
> `CUDA_VISIBLE_DEVICES` a una sola GPU. Se devi vincolare le GPU, usane due,
> es. `export CUDA_VISIBLE_DEVICES=0,1`.

#### Llama-3.1-70B-Instruct

```bash
# === Llama · MuSiQue — forward — 600 domande (200 per hop) ===
tmux new -s llama_musique
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx
bash scripts/final_experiments/llama70b_homer/run_llama_musique_600q_forward.sh \
    2>&1 | tee logs/final_llama_musique_600q_forward.log
```

```bash
# === Llama · NewsQA — forward — 600 domande ===
tmux new -s llama_newsqa
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx
bash scripts/final_experiments/llama70b_homer/run_llama_newsqa_600q_forward.sh \
    2>&1 | tee logs/final_llama_newsqa_600q_forward.log
```

```bash
# === Llama · FictionalQA — forward — 600 domande ===
tmux new -s llama_fictionalqa
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx
bash scripts/final_experiments/llama70b_homer/run_llama_fictionalqa_600q_forward.sh \
    2>&1 | tee logs/final_llama_fictionalqa_600q_forward.log
```

#### gpt-oss-120b

```bash
# === gpt-oss · MuSiQue — forward — 600 domande (200 per hop) ===
tmux new -s gptoss_musique
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx
bash scripts/final_experiments/gptoss120b_homer/run_gptoss_musique_600q_forward.sh \
    2>&1 | tee logs/final_gptoss_musique_600q_forward.log
```

```bash
# === gpt-oss · NewsQA — forward — 600 domande ===
tmux new -s gptoss_newsqa
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx
bash scripts/final_experiments/gptoss120b_homer/run_gptoss_newsqa_600q_forward.sh \
    2>&1 | tee logs/final_gptoss_newsqa_600q_forward.log
```

```bash
# === gpt-oss · FictionalQA — forward — 600 domande ===
tmux new -s gptoss_fictionalqa
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx
bash scripts/final_experiments/gptoss120b_homer/run_gptoss_fictionalqa_600q_forward.sh \
    2>&1 | tee logs/final_gptoss_fictionalqa_600q_forward.log
```

### Fase B — PASSAGGIO chains da Homer a Lisa

L'OFS/recall legge le `rewriting_chains_*.csv` prodotte da Homer. Se Homer e Lisa
**condividono** `results/` sul NAS, salta questo passaggio. Altrimenti, da Lisa:

```bash
# Su LISA — copia le chains forward dal results di Homer
# (adatta host/path se servono; qui assumo rsync su SSH)
mkdir -p ~/Baseline/results/final/llama70b/musique_600q
rsync -av sacchet@homer:~/Baseline/results/final/llama70b/ \
          ~/Baseline/results/final/llama70b/
rsync -av sacchet@homer:~/Baseline/results/final/gptoss120b/ \
          ~/Baseline/results/final/gptoss120b/
```

> Basta che il file `rewriting_chains_<dataset>_600q.csv` esista su Lisa nello
> stesso path `results/final/<model>/<dataset>_600q/`. In alternativa passa il
> path esplicito: `CHAINS=/percorso/chains.csv bash ..._ofs_recall.sh`.

### Fase C — OFS + RECALL su LISA

Gira su Lisa (giudice Gemma-3-4B, una GPU basta). Un blocco tmux per dataset.

#### Llama-3.1-70B

```bash
# === Llama · MuSiQue — OFS + recall ===
tmux new -s llama_musique_ofs
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx; export CUDA_VISIBLE_DEVICES=1
bash scripts/final_experiments/llama70b_homer/run_llama_musique_600q_ofs_recall.sh \
    2>&1 | tee logs/final_llama_musique_600q_ofs_recall.log
```

```bash
# === Llama · NewsQA — OFS + recall ===
tmux new -s llama_newsqa_ofs
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx; export CUDA_VISIBLE_DEVICES=1
bash scripts/final_experiments/llama70b_homer/run_llama_newsqa_600q_ofs_recall.sh \
    2>&1 | tee logs/final_llama_newsqa_600q_ofs_recall.log
```

```bash
# === Llama · FictionalQA — OFS + recall ===
tmux new -s llama_fictionalqa_ofs
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx; export CUDA_VISIBLE_DEVICES=1
bash scripts/final_experiments/llama70b_homer/run_llama_fictionalqa_600q_ofs_recall.sh \
    2>&1 | tee logs/final_llama_fictionalqa_600q_ofs_recall.log
```

#### gpt-oss-120b

```bash
# === gpt-oss · MuSiQue — OFS + recall ===
tmux new -s gptoss_musique_ofs
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx; export CUDA_VISIBLE_DEVICES=1
bash scripts/final_experiments/gptoss120b_homer/run_gptoss_musique_600q_ofs_recall.sh \
    2>&1 | tee logs/final_gptoss_musique_600q_ofs_recall.log
```

```bash
# === gpt-oss · NewsQA — OFS + recall ===
tmux new -s gptoss_newsqa_ofs
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx; export CUDA_VISIBLE_DEVICES=1
bash scripts/final_experiments/gptoss120b_homer/run_gptoss_newsqa_600q_ofs_recall.sh \
    2>&1 | tee logs/final_gptoss_newsqa_600q_ofs_recall.log
```

```bash
# === gpt-oss · FictionalQA — OFS + recall ===
tmux new -s gptoss_fictionalqa_ofs
cd ~/Baseline; source .venv/bin/activate; export HF_TOKEN=hf_xxxx; export CUDA_VISIBLE_DEVICES=1
bash scripts/final_experiments/gptoss120b_homer/run_gptoss_fictionalqa_600q_ofs_recall.sh \
    2>&1 | tee logs/final_gptoss_fictionalqa_600q_ofs_recall.log
```

---

## 3. Riepilogo: quale comando, dove

| Esperimento | Server | Comando |
|---|---|---|
| OLMo · MuSiQue | Lisa | `run_olmo_musique_600q.sh` |
| OLMo · NewsQA | Lisa | `run_olmo_newsqa_600q.sh` |
| OLMo · FictionalQA | Lisa | `run_olmo_fictionalqa_600q.sh` |
| Qwen · MuSiQue | Lisa | `run_qwen_musique_600q.sh` |
| Qwen · NewsQA | Lisa | `run_qwen_newsqa_600q.sh` |
| Qwen · FictionalQA | Lisa | `run_qwen_fictionalqa_600q.sh` |
| Llama · MuSiQue | Homer → Lisa | `run_llama_musique_600q_forward.sh` → `_ofs_recall.sh` |
| Llama · NewsQA | Homer → Lisa | `run_llama_newsqa_600q_forward.sh` → `_ofs_recall.sh` |
| Llama · FictionalQA | Homer → Lisa | `run_llama_fictionalqa_600q_forward.sh` → `_ofs_recall.sh` |
| gpt-oss · MuSiQue | Homer → Lisa | `run_gptoss_musique_600q_forward.sh` → `_ofs_recall.sh` |
| gpt-oss · NewsQA | Homer → Lisa | `run_gptoss_newsqa_600q_forward.sh` → `_ofs_recall.sh` |
| gpt-oss · FictionalQA | Homer → Lisa | `run_gptoss_fictionalqa_600q_forward.sh` → `_ofs_recall.sh` |

I path completi sono `scripts/final_experiments/<cartella_modello>/<script>`.

---

## 4. Smoke test (consigliato prima del run vero)

Riduci il campione con variabili d'ambiente — vale per qualsiasi launcher.

```bash
# MuSiQue: 5 per hop = 15 domande
N_PER_HOP=5 bash scripts/final_experiments/olmo32b_lisa/run_olmo_musique_600q.sh

# NewsQA / FictionalQA: 10 domande
N_ITEMS=10 bash scripts/final_experiments/qwen30b_lisa/run_qwen_newsqa_600q.sh

# Homer: smoke del solo forward
N_PER_HOP=5 SKIP_PPL=1 \
  bash scripts/final_experiments/llama70b_homer/run_llama_musique_600q_forward.sh
```

---

## 5. Output prodotti

Tutto finisce in `results/final/<model>/<dataset>_600q/`:

```
rewriting_chains_<dataset>_600q.csv                       # chains + n_tokens (token length)
rewriting_chains_<dataset>_600q_answer_f1.csv             # Answer F1
rewriting_chains_<dataset>_600q_bertscore.csv             # BERTScore Baseline + Consecutive
rewriting_chains_<dataset>_600q_bleurt.csv                # BLEURT baseline + consecutive + answer
rewriting_chains_<dataset>_600q_perplexity.csv            # Perplexity
rewriting_chains_<dataset>_600q_openfactscore.csv         # FactScore (precision)
rewriting_chains_<dataset>_600q_openfactscore_details.csv # fatti atomici E_0 (riusati dal recall)
rewriting_chains_<dataset>_600q_openfactscore_recall.csv  # Recall (musique / fictionalqa, giudice Gemma)
rewriting_chains_<dataset>_600q_recall_nli.csv            # Recall (newsqa, NLI cross-encoder)
```

Controllo veloce di avanzamento:

```bash
wc -l results/final/olmo32b/musique_600q/*.csv
```

---

## 6. Variabili d'ambiente utili (override opzionali)

Valgono su tutti i launcher (hanno default sensati già impostati):

| Variabile | Default | Cosa fa |
|---|---|---|
| `N_PER_HOP` | 200 | domande per hop (solo MuSiQue) → ×3 = totale |
| `N_ITEMS` | 600 | domande totali (NewsQA / FictionalQA) |
| `N_ITERATIONS` | 3 | step di rewriting iterativo |
| `MAX_NEW_TOKENS` | per-dataset | lunghezza max output rewrite |
| `MAX_MODEL_LEN` | per-dataset | context window vLLM |
| `TP` | 1 (Lisa) / 2 (Homer) | tensor-parallel size |
| `BACKEND` | per-modello | `vllm` \| `hf` \| `""` (vedi README) |
| `QUANT` | per-modello | es. `bitsandbytes` per NF4 4-bit |
| `SKIP_PPL` | (off) | `SKIP_PPL=1` salta la perplexity |
| `OUT_DIR` | `results/final/...` | cartella output |
| `HF_TOKEN` | — | **obbligatorio** (modelli gated) |

Dettagli architetturali (backend per modello, perché i due script su Homer,
introspezione dei flag) in [scripts/final_experiments/README.md](scripts/final_experiments/README.md).
```
