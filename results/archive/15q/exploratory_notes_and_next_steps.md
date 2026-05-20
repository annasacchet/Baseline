# Note esplorative e prossimi passi

_Last updated: 2026-05-09_

Questo documento riassume due piste di analisi qualitativa rimaste aperte sul pilot 15q, introduce i due nuovi dataset (NewsQA e FictionalQA) per cui è già pronta la pipeline, e raccoglie l'evidenza che fa pensare a un effetto **quantizzazione** sui run 300q (Lisa, 4-bit) rispetto al pilot 15q (bf16).

---

## 1. Investigazioni esplorative ancora aperte (15q)

Le analisi che seguono sono **qualitative**: il pilot ha 15 domande × 4 istruzioni × 3 run = 180 chain. È una numerosità troppo piccola per concludere statisticamente — i test condotti finora ([investigations_sign_tests.csv](investigations_sign_tests.csv)) sono per la maggior parte underpowered. Le piste qui sotto vanno lette come spunti di interpretazione, da rivalidare quantitativamente sui run 300q (e poi su NewsQA / FictionalQA).

### 1.1 LLM-as-a-judge per capire `elaborate`

`elaborate` è l'unica istruzione che mostra una doppia degradazione monotona: Answer F1 da 0.706 a 0.602 e OpenFactScore da 0.872 a 0.807 da step 1 a step 3 (vedi [data_analysis_notes.md](data_analysis_notes.md), §2.1 e §6.1). L'ipotesi del brainstorming era che `elaborate` introducesse **allucinazioni cumulative**: dettagli aggiuntivi non ancorati a E₀ che diluiscono o sovrascrivono i fatti supportanti.

**Cosa abbiamo già.** Lo script [scripts/15q/elaborate_gpt_analysis.py](../../scripts/15q/elaborate_gpt_analysis.py) usa `gpt-4o-mini` come judge sui pair (orig, rewritten) di tutte le 45 chain `elaborate`, sui confronti consecutivi (0→1, 1→2, 2→3) e overall (0→3). Per ogni confronto produce:

- `analysis` — commento libero in italiano
- `lost` — lista di fatti scomparsi/distorti
- `added` — lista di fatti introdotti non grounded in E₀

I risultati sono in [results/15q/elaborate_gpt_analysis.csv](elaborate_gpt_analysis.csv) (180 righe) e [.md](elaborate_gpt_analysis.md).

**Cosa abbiamo provato a misurare.** Correlazione tra `n_added` per step e drop di Answer F1 nello stesso step (Spearman ρ = −0.002, p = 0.98 su 135 confronti consecutivi). Friedman su `n_added` step-by-step: media 3.42 → 3.04 → 3.09, p = 0.08 — andamento leggermente decrescente, non crescente.

**Come leggerlo.** La metrica `n_added` totale è troppo grezza: cinque aggiunte irrilevanti su personaggi laterali non spostano F1, ma una sola distorsione del fatto-bersaglio sì. Esempio dal report (qid `2hop__14092_8311`, gold = "1216"): la chain aggiunge frasi laterali come "King John's complex marital life saw him marry twice" (innocua per la domanda) e in parallelo distorce "John had five legitimate children, all by Isabella" (rompe il link Henry III → primogenito di Isabella, decisivo per la domanda). `n_added` le conta entrambe come +1.

**Prossimo passo qualitativo.** Filtrare le liste `added` cercando i token della gold answer o delle supporting facts MuSiQue, e correlare *quel* count (non `n_added` totale) con ΔF1. Lo spot-check manuale di 5–10 confronti dove F1 crolla sembra utile per capire se l'allucinazione è **conflittuale** (introduce dati che il QA model preferisce alla risposta corretta) o solo **diluitiva**.

### 1.2 Le 96 chain con F1 = 0 a step 1

Su 180 chain, **96 hanno F1=0 al primo step** e sono state escluse dalle analisi aggregate. La ragione tecnica è il floor effect: non si può misurare un drop quando il punto di partenza è già zero. Ma è ~53% del dataset, vale la pena guardarlo.

**Cosa succede dopo lo zero al primo step?** Sui 96 casi:

| Trajectory                         | Conteggio | Note                                                |
|------------------------------------|-----------|-----------------------------------------------------|
| Stayed zero (F1=0 anche step 2/3)  | 85        | Il rewriting non ha mai esposto il fatto al QA      |
| Late recovery (F1>0 a step 3)      | 8         | Una riformulazione successiva ha sbloccato qualcosa |
| Transient recovery (solo a step 2) | 3         | Sblocco passeggero, poi riperso                     |

Distribuzione delle 11 recovery per istruzione: paraphrase 4, shorten 4, elaborate 3, formality 0. Curioso che `formality` non recuperi mai — ma con n=11 è solo un'osservazione qualitativa, non un risultato (chi-square per uniformità: p = 0.27).

**Tre meccanismi possibili per il "late recovery".**

1. Riformulazione che il QA model parsea meglio (sensibilità allo stile, caso 3 di [data_analysis_notes.md](data_analysis_notes.md) §7).
2. Cambio di posizione del fatto nel testo che basta al QA model.
3. Caso (apparente) — falso negativo della normalizzazione SQuAD-style: la pred a step 1 era già semanticamente corretta ma scritta diversamente dal gold. Questo è un caso noto: gold="1,400 years" vs pred="1400 years ago" → F1=0.8 non 1.0.

**Perché vale la pena guardarli a mano.** Esempio concreto: `3hop1__305566_568433_47686 / paraphrase / run 0` — gold "Toronto Coach Terminal", pred a step 1 "Rodoferroviária station" (F1=0), pred a step 3 "Toronto Coach Terminal" (F1=1.0). Cos'è cambiato tra E₁ ed E₃? Spot-check utile per capire se è il rewriting a esporre il fatto, o il QA model a "vederlo" meglio in una formulazione diversa.

**Refusal vs WRONG vs EMPTY.** Sui 96 predicted_answer:
- ~36 rifiuti espliciti ("The context does not provide…")
- ~56 risposte sbagliate
- ~4 vuoti

Distribuzione per istruzione molto piatta — chi-square: p=0.78, Cramer's V=0.13. Su 15q non c'è segnale di "shorten causa più rifiuti del previsto".

**Plan operativo (pulito, da rifare sui 300q).**
1. Etichettare predicted_answer in `WRONG | REFUSAL | EMPTY` con regex.
2. Per i `WRONG`, usare `gpt-4o-mini` come judge per verificare equivalenza semantica con il gold (template clonabile da [scripts/15q/recall_gpt_eval.py](../../scripts/15q/recall_gpt_eval.py)). Recupera i falsi negativi della normalizzazione.
3. Spot-check manuale su 5-10 chain "stayed_zero" per capire da dove viene la risposta sbagliata (distrattore? memoria parametrica? fatto distorto?).

---

## 2. Due nuovi dataset: NewsQA e FictionalQA

La pipeline di rewriting è stata estesa a due dataset complementari a MuSiQue. Sono già pronti gli script per (i) generare le chain di rewriting, (ii) calcolare Answer F1, (iii) calcolare OpenFactScore. Tutta la pipeline è stata **validata su `gpt-4o-mini`** (smoke test): la generazione gira, gli output sono coerenti, gli evaluator restituiscono numeri sensati. **Manca solo il run completo su GPU server** — in attesa di slot libero su Homer/Lisa/Bart.

### 2.1 NewsQA (Maluuba)

- **Sorgente.** `combined-newsqa-data-v1.csv` — articoli CNN con domande crowdsourced.
- **Formato.** Una (story, question) per chain. Per ogni story si seleziona la *miglior domanda answerable* (almeno uno span validato concordato da ≥2 crowdworker, escludendo `is_answer_absent` e `is_question_bad`).
- **Gold answer.** Tutti gli span validati sono mantenuti come alias (`||`-joined). L'F1 evaluator prende il max su questo set, replicando la metrica ufficiale NewsQA / SQuAD.
- **Differenza chiave da MuSiQue.** Domande **single-hop** estrattive su articoli giornalistici reali. Niente ragionamento multi-hop, niente distrattori multipli. Permette di isolare il segnale di degradazione dal rumore multi-hop che domina su MuSiQue.
- **Script.**
  - [scripts/newsqa/rewriting_pipeline_newsqa.py](../../scripts/newsqa/rewriting_pipeline_newsqa.py)
  - [scripts/newsqa/answer_f1_eval_newsqa.py](../../scripts/newsqa/answer_f1_eval_newsqa.py)
  - [scripts/newsqa/openfactscore_eval_newsqa.py](../../scripts/newsqa/openfactscore_eval_newsqa.py)

### 2.2 FictionalQA

- **Sorgente.** `jwkirchenbauer/fictionalqa` su HuggingFace Hub — eventi *fittizi* generati per blocco totale del knowledge leakage del QA model.
- **Formato.** 5 stili (blog, news, …) × 300 eventi = 1500 documenti. Per ogni fiction si seleziona la miglior domanda answerable filtrando con `grade_blind == 0` (LLM non risponde senza il documento) e `grade_informed == 1` (LLM risponde con il documento).
- **Differenza chiave.** Il fatto è **fittizio** → il QA model non può rispondere da memoria parametrica. Ogni F1>0 misura genuinamente la preservazione del fatto attraverso il rewriting. Risolve il problema noto del MuSiQue: parte degli F1 a step 0 potrebbe essere prior knowledge, non comprensione del testo.
- **Script.**
  - [scripts/fictionalqa/rewriting_pipeline_fictionalqa.py](../../scripts/fictionalqa/rewriting_pipeline_fictionalqa.py)
  - [scripts/fictionalqa/answer_f1_eval_fictionalqa.py](../../scripts/fictionalqa/answer_f1_eval_fictionalqa.py)
  - [scripts/fictionalqa/openfactscore_eval_fictionalqa.py](../../scripts/fictionalqa/openfactscore_eval_fictionalqa.py)

### 2.3 A cosa servono

- **NewsQA** — controllo: la degradazione che vediamo su MuSiQue dipende dalla difficoltà multi-hop, o si replica anche su QA estrattivo single-hop su testo reale?
- **FictionalQA** — controllo più stringente: la degradazione misurata da F1 dipende davvero dal rewriting, o parte è prior knowledge del QA model? Se OFS scende anche qui, è un segnale pulito di hallucination indipendente dal QA.

Insieme, i tre dataset (MuSiQue + NewsQA + FictionalQA) coprono: multi-hop con distrattori (MuSiQue) → single-hop estrattivo reale (NewsQA) → single-hop su fatti fittizi (FictionalQA), permettendo di triangolare il fenomeno.

### 2.4 Stato attuale

- Pipeline scritta, tipi e schema CSV coerenti tra i tre dataset.
- Smoke test passato con `gpt-4o-mini` come backbone (rewriting + F1 + OFS).
- I run veri vanno fatti con il modello target (OLMo-3.1-32B-Instruct per coerenza con MuSiQue).
- **Bloccante: GPU.** I run su Lisa/Homer/Bart sono già occupati con i 300q (vedi memoria `project_300q_runs`). Lo step successivo è schedulare i run NewsQA e FictionalQA non appena uno dei nodi si libera.

---

## 3. 300q: il ruolo della quantizzazione

I run 300q stanno girando su Lisa in **4-bit NF4** (vincolo VRAM 3090 24GB), mentre il pilot 15q era stato fatto in **bfloat16** su Homer. Questa è una differenza non banale per il rewriting, e i numeri lo confermano. Sui **10 qid in overlap** tra 15q e 300q (stessa domanda, stesso prompt, stesso modello, ma diverso dtype), i pattern divergono.

### 3.0 Verifica: i qid in overlap hanno davvero la stessa E₀

Prima di leggere i numeri, ho verificato che lo stesso qid in 15q e 300q condivida byte-per-byte la E₀ e la domanda. Tutti i 10 qid in overlap passano:

- **E₀ identica** — md5 uguale per tutti i 10 testi a step 0 (verifica eseguita su `text` e su `question`).
- **Differenza in `n_tokens` non strutturale** — sullo stesso testo, 15q riporta in media ~50 token in più di 300q (es. 2038 vs 1995 per `2hop__14092_8311`). È una differenza di tokenizer/versione `transformers`, non di contenuto.
- **Comportamento di rewriting nettamente diverso** — esempio per `2hop__14092_8311 / elaborate / run 0`, step 1: bf16 produce 5975 caratteri (1388 token), 4-bit produce 3642 caratteri (851 token), entrambi a partire dalla stessa identica E₀.

L'unica variabile cambiata tra i due run è il dtype del modello di rewriting.

### 3.1 Token count — `elaborate` smette di elaborare in 4-bit

Lunghezza media (token) sui 10 qid in overlap, per step × istruzione:

**15q (bfloat16, Homer)**

| Instruction | Step 0 | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|--------|
| elaborate   | 2360   | 1499   | 1548   | **1621** |
| formality   | 2360   | 1316   | 1252   | 1216   |
| paraphrase  | 2360   |  890   |  832   |  796   |
| shorten     | 2360   |  574   |  478   |  439   |

**300q (4-bit NF4, Lisa)**

| Instruction | Step 0 | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|--------|
| elaborate   | 2308   |  577   |  523   | **538**  |
| formality   | 2308   |  871   |  705   |  609   |
| paraphrase  | 2308   |  492   |  384   |  335   |
| shorten     | 2308   |  432   |  332   |  285   |

Confronto chiave: `elaborate` in bf16 segue l'istruzione e *cresce* da step 1 in poi (1499 → 1621). In 4-bit *comprime* (577 → 538) — più di quanto comprima `paraphrase` o `formality` in bf16. Tutte le istruzioni convergono verso un comportamento di compressione, indipendentemente dal contenuto del prompt.

**Test indipendente con `gpt-4o-mini`.** Ho lanciato qualche prompt di rewriting su `gpt-4o-mini` con istruzione `elaborate` su gli stessi testi: il modello *non* comprime così aggressivamente. Questo riduce ulteriormente la probabilità che la compressione sia un artefatto del prompt o della pipeline; punta direttamente al modello quantizzato come responsabile.

### 3.2 Answer F1 — degrada di più sui 300q (overlap qid)

Sui 10 qid in overlap, F1 medio per step:

**15q (bfloat16)**

| Instruction | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|
| elaborate   | 0.283  | 0.332  | 0.347  |
| formality   | 0.312  | 0.287  | 0.287  |
| paraphrase  | 0.287  | 0.326  | 0.348  |
| shorten     | 0.320  | 0.254  | 0.291  |

**300q (4-bit)**

| Instruction | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|
| elaborate   | 0.274  | 0.188  | 0.181  |
| formality   | 0.311  | 0.265  | 0.280  |
| paraphrase  | 0.260  | 0.224  | 0.226  |
| shorten     | 0.253  | 0.234  | 0.226  |

Le **stesse domande**, sullo stesso modello, perdono molto più F1 quando il rewriting è fatto in 4-bit. Il pattern è coerente con l'osservazione sui token: il modello quantizzato comprime di più → rimuove più fatti → F1 cala di più.

### 3.3 Implicazione

Quando i numeri 300q saranno presentati come "trend di degradazione", c'è il rischio reale che il segnale sia in parte *artefatto della quantizzazione* del modello di rewriting, non degradazione fattuale intrinseca del rewriting iterativo. Le evidenze:

1. **Comportamento divergente di `elaborate`** (espande in bf16, comprime in 4-bit) sugli stessi qid.
2. **F1 più basso in 4-bit** sugli stessi qid, anche per istruzioni dove la compressione non è richiesta.
3. **`gpt-4o-mini` non riproduce la compressione** sotto `elaborate` con gli stessi prompt.

**Cosa serve per chiudere la questione.** Un run di confronto controllato: una manciata di qid in `bf16` *vs* `4-bit` con stesso modello, stesso prompt, stesso seed. Se l'effetto si replica, va dichiarato come confound nei report 300q e bisogna scegliere — o si rilanciano i 300q in bf16 (anche solo su Homer, accettando i tempi più lunghi), o si tiene la 4-bit ma si quantifica e dichiara il bias.

---

## 4. Cosa fare next

Priorità ordinata:

1. **Run di controllo bf16 vs 4-bit** su 5–10 qid (priorità alta — sblocca l'interpretazione di tutti i numeri 300q).
2. **NewsQA + FictionalQA** in coda non appena una GPU si libera. Pipeline pronta.
3. **Spot-check qualitativo** sulle chain "late recovery" del 15q (basso costo, alta resa interpretativa).
4. **LLM-as-judge mirato** sul fatto-bersaglio (non `n_added` totale) per `elaborate` — su 300q dove n è abbastanza grande per misurare correlazioni piccole.
