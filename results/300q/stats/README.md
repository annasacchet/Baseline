# 300q — Risultati statistici spiegati

Questo documento spiega in modo divulgativo cosa è stato calcolato dallo script
[`scripts/300q/inference_tests.py`](../../../scripts/300q/inference_tests.py) e
come leggere i numeri.

> **Nota importante:** sul run 300q **l'OpenFActScore (OFS) è stato calcolato
> solo sulle 55 domande 2-hop** (su 297 totali). Per non veicolare conclusioni
> non generalizzabili, **in questo README OFS è escluso**: tutte le analisi qui
> sotto si basano su **Answer F1** e **BERTScore**, che invece sono bilanciati
> su tutti i livelli di complessità. Vedi §6 (Limitazioni).

---

## 1. Cosa stiamo misurando

Per ogni domanda di MuSiQue (297 in totale), il testo di supporto viene riscritto
**3 volte di fila** (step 1, 2, 3) sotto **4 tipi di istruzione**:

- **content** → `elaborate` / `shorten` (cambiare la quantità di contenuto)
- **style** → `formality` / `paraphrase` (cambiare lo stile mantenendo il contenuto)

Ogni rewriting è ripetuto **3 volte** (run 0, 1, 2) per misurare la stocasticità.
Sul testo di ogni step misuriamo due metriche (analizzate qui):

| Metrica | Cosa misura | Range | Step disponibili |
|---|---|---|---|
| **Answer F1** | il modello QA risponde correttamente? | 0–1 | 0, 1, 2, 3 |
| **BERTScore baseline** | quanto il rewriting somiglia all'originale (step 0) | 0–1 | 1, 2, 3 |
| **BERTScore consecutive** | quanto lo step k somiglia allo step k-1 | 0–1 | 1, 2, 3 |

> **Step 0** = testo originale, non riscritto. È il *baseline* da cui partiamo.

---

## 2. Glossario veloce (statistica per chi non se ne occupa)

- **p-value**: probabilità di osservare questo risultato (o uno più estremo) se in
  realtà *non* ci fosse nessun effetto. **Se p < 0.05, l'effetto è considerato
  "statisticamente significativo"**. Più p è piccolo, più siamo sicuri che
  l'effetto esista.
- **p_holm**: il p-value corretto per il fatto che facciamo più test in parallelo
  (Holm è una correzione standard). Si guarda **questo**, non il p grezzo.
- **mean_diff**: di quanto cambia la metrica in media tra due step (es. −0.17 =
  cala di 17 punti percentuali).
- **effect size / rank_biserial**: quanto è *grande* l'effetto, indipendentemente
  dal numero di osservazioni. Vicino a 0 = effetto trascurabile; vicino a 1 =
  effetto massiccio.
- **ICC (Intraclass Correlation)**: percentuale di varianza spiegata da un
  raggruppamento. Es. ICC(qid) = 0.4 vuol dire che **il 40% della variabilità è
  dovuto a "quale domanda è"** — alcune domande sono semplicemente più facili di
  altre.
- **Random intercept `(1|qid)`**: nei modelli misti diciamo "tieni conto che la
  stessa domanda è stata vista più volte, non trattare le osservazioni come
  indipendenti". Senza questo, i p-value sarebbero artificialmente piccoli.

---

## 3. Quali test abbiamo fatto e perché

| Test | Domanda | Modello |
|---|---|---|
| **1** | Step degrada Answer F1? | GLMM logistico su `F1>0`, contrasti pianificati |
| **2** | Step muove BERTScore? | LMM su baseline e consecutive |
| **3** | L'istruzione (elaborate/shorten/formality/paraphrase) cambia il pattern? | Modelli separati per content/style |
| **4** | content vs style hanno traiettorie diverse? | LMM con interazione step × group |
| **5** | I 3 run aggiungono variabilità che dobbiamo modellare? | Decomposizione varianza + Friedman |
| **6** | F1 e BERTScore correlano tra loro? | LMM e repeated-measures correlation |
| **7** | La complessità (n-hop) influenza il degrado? (RQ2b) | LMM con interazione step × n_hop |
| **8** | La lunghezza del rewriting (n_tokens) conta? | Descrittiva + mediation analysis |
| **9** | I risultati reggono con test non parametrici? | Friedman omnibus |

Tutti i test usano `(1|qid)` per tenere conto del fatto che ogni domanda è osservata
sotto tutte le condizioni.

---

## 4. Risultati

### 4.1 Answer F1 cala con gli step

**File:** [`inference/f1_step_contrasts.csv`](inference/f1_step_contrasts.csv)

| Contrasto | Differenza media | p_holm | Effect size |
|---|---|---|---|
| step 0 → 1 | **−0.169** | **1.1e-08** | 0.38 (medio) |
| step 1 → 2 | −0.030 | 0.93 (n.s.) | 0.02 |
| step 2 → 3 | −0.017 | 0.93 (n.s.) | 0.04 |

**Lettura:** Il vero crollo della capacità di rispondere correttamente avviene
**dopo il primo rewriting**: l'F1 scende di ~17 punti percentuali. Gli step
successivi (1→2 e 2→3) non producono un calo statisticamente significativo
*sopra* a quello del primo step. In altre parole: **già un solo rewriting fa la
maggior parte del danno**.

> Diagnostica importante: l'F1 ha distribuzione *bimodale* (molti 0 e 1), per cui
> abbiamo usato un GLMM logistico sulla versione binarizzata (`F1>0`).
> La conclusione qualitativa non cambia se modelliamo l'F1 continua.

### 4.2 BERTScore — due dinamiche opposte

**File:** [`inference/bs_baseline_step_contrasts.csv`](inference/bs_baseline_step_contrasts.csv) e [`inference/bs_consecutive_step_contrasts.csv`](inference/bs_consecutive_step_contrasts.csv)

**BERTScore baseline** (similarità vs testo originale):

| Contrasto | Differenza media | p_holm |
|---|---|---|
| step 1 → 2 | −0.008 | 5.1e-49 |
| step 2 → 3 | −0.004 | 1.5e-46 |

→ Il testo si allontana dall'originale **monotonicamente**, ma di pochissimo (1
punto percentuale totale).

**BERTScore consecutive** (similarità step k vs step k-1):

| Contrasto | Differenza media | p_holm |
|---|---|---|
| step 1 → 2 | **+0.085** | 3.8e-50 |
| step 2 → 3 | **+0.015** | 1.5e-47 |

→ Gli step diventano **sempre più simili al precedente**: il rewriting *converge*
a un attrattore. Dopo il primo step, il modello cambia sempre meno.

**Insieme**, questi due numeri raccontano una storia precisa: il sistema si
allontana dall'originale soprattutto al primo step, e poi si stabilizza intorno
a una nuova versione "del modello".

### 4.3 L'istruzione cambia il pattern?

**File:** `inference/{f1,bs_base,bs_cons}_lmm_{content,style}_step_x_instr.csv`

Per ciascun gruppo (content/style) abbiamo confrontato le due istruzioni.

- **F1** — Le 4 istruzioni hanno traiettorie quasi sovrapposte: `formality` è
  leggermente più "robusto" (+1–2 punti F1 a step 3) ma le differenze sono
  piccole e non significative. **L'istruzione conta poco per F1.**
- **BERTScore** — Differenze grandi su `paraphrase` (che riscrive di più,
  quindi BERT cala). Coerente con l'intento dell'istruzione.

### 4.4 content vs style hanno traiettorie diverse?

**File:** [`inference/f1_logit_step_x_group.csv`](inference/f1_logit_step_x_group.csv) e analoghi

- **Su F1**: nessuna interazione significativa (p_interaction ≥ 0.2). Content e
  style degradano in modo simile.
- **Su BERTScore baseline**: style è 1.9 punti più alto in media (più simile
  all'originale), ma la *forma* del declino è uguale.
- **Su BERTScore consecutive**: interazione **significativa** (p ≈ 1e-14):
  **content si stabilizza più velocemente di style**. Cioè: riscritture di
  contenuto convergono prima a un attrattore; quelle di stile continuano un po'
  di più a cambiare.

### 4.5 Le 3 run aggiungono variabilità?

**File:** [`inference/run_variance_decomposition.csv`](inference/run_variance_decomposition.csv)

| Metrica | ICC(qid) | ICC(run) | Friedman p |
|---|---|---|---|
| F1 | 43% | **3.1%** | 0.028 |
| BERT base | 43% | **5.6%** | 7.7e-09 |
| BERT cons | 7.5% | **2.2%** | 0.007 |

**Lettura:** la variabilità tra run è **piccola** (2–6%): quasi tutta la varianza
viene dalle differenze tra domande (ICC qid ~40%) e dal residuo. Il p-value
Friedman è significativo per F1/BERT solo perché abbiamo 297 osservazioni —
l'*effetto pratico* di run è trascurabile, e in molti casi puoi mediare le 3 run
senza perdere informazione importante.

> **Caveat:** "non significativo praticamente" ≠ "non esiste". Se vuoi fare claim
> sulla singola run (es. "la run 2 è migliore della run 0"), allora la
> variabilità run è grande abbastanza da essere rilevata.

### 4.6 F1 e BERTScore correlano tra loro?

**File:** [`inference/corr_repeated_measures.csv`](inference/corr_repeated_measures.csv)

Correlazione *entro qid* (controllando per il fatto che alcune domande sono
sistematicamente più facili):

| Coppia | r | 95% CI | p |
|---|---|---|---|
| F1 ↔ BERTScore base | **0.193** | [0.15, 0.24] | 2e-17 |

**Lettura:** correlazione positiva ma **debole**. La similarità lessicale
all'originale è un debole proxy per "il sistema risponde meglio" — la
relazione c'è ma non è abbastanza forte per predire F1 da BERTScore.

### 4.7 RQ2b — La complessità della domanda (n-hop) influenza il degrado?

**File:** `inference/*_step_x_hop.csv` e `*_degradation_by_hop.csv`

L'Answer F1 e il BERTScore sono bilanciati su 2-hop (100), 3-hop (97), 4-hop
(100), quindi possiamo testare RQ2b su queste metriche.

**Answer F1 — medie per hop × step** (file [`f1_logit_step_x_hop.csv`](inference/f1_logit_step_x_hop.csv)):

| n_hop | step 0 | step 1 | step 2 | step 3 | calo 0→3 |
|---|---|---|---|---|---|
| 2 | 0.530 | 0.367 | 0.317 | 0.297 | −0.233 |
| 3 | 0.526 | 0.335 | 0.316 | 0.294 | −0.232 |
| 4 | 0.430 | 0.278 | 0.258 | 0.248 | −0.182 |

Cosa si vede:
- Le **4-hop partono già più basse** anche sul testo originale (0.43 vs 0.53 di
  2/3-hop) — coerente: domande più complesse sono più difficili da rispondere.
- Il **calo assoluto** è simile o leggermente minore per 4-hop (−0.18 vs −0.23):
  significa che le 4-hop **degradano un po' meno in valore assoluto**, ma
  partono da più in basso. In percentuale, perdono di più rispetto al baseline.
- L'interazione `step × hop` è significativa per 4-hop a step 2 e 3
  (p = 0.042 e 0.010): le 4-hop hanno una traiettoria di degrado leggermente
  diversa, *meno ripida in valore assoluto*.

**BERTScore baseline — medie per hop × step**:

| n_hop | step 1 | step 2 | step 3 |
|---|---|---|---|
| 2 | 0.857 | 0.847 | 0.843 |
| 3 | 0.855 | 0.847 | 0.842 |
| 4 | 0.849 | 0.841 | 0.837 |

- Le 4-hop hanno BERTScore leggermente più basso (β = −0.008, p = 0.04): il
  rewriting si allontana un filo di più dall'originale per domande complesse.
- La velocità di drift è la stessa: nessuna interazione significativa.

**BERTScore consecutive — convergenza più rapida per 4-hop**:

| n_hop | step 1 | step 2 | step 3 |
|---|---|---|---|
| 2 | 0.857 | 0.936 | 0.953 |
| 3 | 0.855 | 0.937 | 0.951 |
| 4 | 0.849 | **0.940** | 0.953 |

- Le 4-hop **convergono più velocemente all'attrattore** (interazione step 2 ×
  hop 4: +0.012, p < 1e-9). Cioè: per domande complesse, il modello si stabilizza
  in una nuova versione *prima*.

**Take-away RQ2b (su F1 e BERTScore):**

> Sull'Answer F1, la complessità (n-hop) ha un effetto principale chiaro: le
> 4-hop sono più difficili in assoluto (parto più basse, perdono ~18 punti vs ~23
> di 2/3-hop). Il degrado *step-by-step* è simile in tutte le complessità.
> Sul drift testuale (BERTScore), le 4-hop si allontanano leggermente di più
> dall'originale e convergono più velocemente al loro attrattore. **L'ipotesi
> "domande complesse degradano di più" è supportata in modo modesto su queste
> due metriche. La parte di RQ2b relativa alla fattualità (OFS) resta aperta
> per i limiti sui dati (vedi §6).**

### 4.8 La lunghezza del rewriting (n_tokens) conta? ★

Il numero di token prodotti dal rewriting è una variabile chiave che fino a
qui non avevamo considerato. La domanda è doppia: (a) come si distribuisce
n_tokens fra le istruzioni? (b) la lunghezza spiega parte di quello che
attribuiamo all'istruzione?

#### 4.8.1 Descrittiva: lunghezze per istruzione × step

**File:** [`inference/length_descriptive.csv`](inference/length_descriptive.csv)
e [`inference/length_compliance.csv`](inference/length_compliance.csv)

Il testo originale ha mediana ~2340 token. Dopo il primo rewriting:

| Istruzione | step 1 | step 2 | step 3 | % corte (<200) step 1 | % saturate (≥2048) step 1 |
|---|---|---|---|---|---|
| elaborate | 710 | 647 | 620 | **32.1%** | 1.1% |
| formality | 935 | 722 | 631 | 10.1% | 1.0% |
| paraphrase | 555 | 420 | 357 | **29.7%** | 0.2% |
| shorten | 458 | 352 | 308 | 17.7% | 0.0% |

**Cosa salta all'occhio:**
- **Nessuna istruzione mantiene la lunghezza dell'originale.** Tutte
  comprimono. `formality` è quella che comprime di meno al primo step
  (~38% dell'originale); `shorten` è quella che comprime di più (~19%).
- **`elaborate` è la più aggressiva sul collasso**: 32% delle catene scende
  sotto i 200 token al primo step (vedi
  [`stats/elaborate/`](elaborate/) per l'analisi dedicata).
- **Paraphrase compete con elaborate** sul collasso (29.7% catene corte).
- **Saturazione del cap 2048 trascurabile** sul 300q (≤1.1% in ogni
  condizione), diversamente dal 15q.

#### 4.8.2 La lunghezza predice F1 e BERTScore?

**Modello F1**
[`inference/length_f1_on_logtokens.csv`](inference/length_f1_on_logtokens.csv):
`F1 ~ log(n_tokens) + step + (1|qid)`

| Termine | Coef | p |
|---|---|---|
| log(n_tokens) | **+0.038** | <0.001 |
| step 1 | −0.085 | <0.001 |
| step 2 | −0.104 | <0.001 |
| step 3 | −0.114 | <0.001 |

→ Per ogni *raddoppio* della lunghezza prodotta, F1 cresce di circa
**+0.026 punti**. Più informazione nel testo → più chance di rispondere.

**Modello BERTScore baseline**
[`inference/length_bs_on_logtokens.csv`](inference/length_bs_on_logtokens.csv):

| Termine | Coef | p |
|---|---|---|
| log(n_tokens) | **+0.029** | <0.001 |
| step 2 | −0.004 | <0.001 |
| step 3 | −0.005 | <0.001 |

→ Effetto enorme: gran parte della varianza di BERTScore baseline è
spiegata semplicemente dalla lunghezza. Testi più lunghi mantengono più
sovrapposizione lessicale con l'originale.

#### 4.8.3 Mediation — l'effetto dell'istruzione si attenua controllando per n_tokens?

**Su F1** (file
[`length_f1_mediation_no_tokens.csv`](inference/length_f1_mediation_no_tokens.csv) e
[`length_f1_mediation_with_tokens.csv`](inference/length_f1_mediation_with_tokens.csv)):

Coefficienti rispetto a `elaborate` (baseline):

| Termine | Senza n_tokens | Con n_tokens |
|---|---|---|
| formality vs elaborate | +0.017 (p = 0.02) | +0.003 (p = 0.66) |
| paraphrase vs elaborate | +0.008 (p = 0.25) | **+0.026 (p < 0.001)** |
| shorten vs elaborate | −0.003 (p = 0.71) | **+0.018 (p = 0.02)** |
| log(n_tokens) | — | **+0.047 (p < 0.001)** |

**Sorpresa interessante:** controllando per la lunghezza, il quadro si
**inverte**. Senza n_tokens, formality sembrava la migliore; con n_tokens,
paraphrase e shorten emergono come migliori di elaborate **a parità di
lunghezza**. Vuol dire che a parità di token prodotti, paraphrase e shorten
producono testi più "utili" per QA di elaborate. La superficiale superiorità
di formality era in larga parte dovuta al fatto che formality produce testi
più lunghi.

**Su BERTScore baseline** (file
[`length_bs_mediation_no_tokens.csv`](inference/length_bs_mediation_no_tokens.csv) e
[`length_bs_mediation_with_tokens.csv`](inference/length_bs_mediation_with_tokens.csv)):

| Termine | Senza n_tokens | Con n_tokens |
|---|---|---|
| formality vs elaborate | +0.033 (p < 0.001) | +0.025 (p < 0.001) |
| paraphrase vs elaborate | −0.000 (p = 0.99) | **+0.010 (p < 0.001)** |
| shorten vs elaborate | −0.003 (p < 0.001) | **+0.009 (p < 0.001)** |
| log(n_tokens) | — | **+0.026 (p < 0.001)** |

→ Stesso pattern. Senza n_tokens, paraphrase e shorten sembrano "peggiori"
di elaborate sulla similarità all'originale. Con n_tokens, a parità di
lunghezza sono **migliori**. La lunghezza media più alta di elaborate
mascherava la sua minore qualità lessicale.

#### 4.8.4 Take-away sulla lunghezza

> La lunghezza è un **confondente importante** che non si può ignorare. Tutte
> e quattro le istruzioni comprimono il testo originale, ma in misure
> diverse. Quando si controlla per n_tokens nel modello:
>
> - L'effetto di `step` sull'F1 si riduce di ~30% (da −0.114 a −0.080 a step
>   3): parte del degrado per step è attribuibile alla compressione
>   progressiva, non al rewriting "in sé".
> - L'ordine relativo tra istruzioni cambia: paraphrase e shorten emergono
>   come **superiori a elaborate a parità di lunghezza** — risultato opposto
>   alla lettura senza n_tokens.
> - La lunghezza spiega da sola gran parte di BERTScore baseline (coef
>   log_ntokens = +0.029, dominante rispetto a step).
>
> **Implicazione per il paper:** confronti tra istruzioni o tra step senza
> aggiustare per n_tokens **non sono interpretabili** in modo netto. Per
> conclusioni causali sul rewriting bisogna controllare per la lunghezza
> prodotta.

### 4.9 Robustness — i test non parametrici confermano?

**File:** [`inference/robustness_friedman.csv`](inference/robustness_friedman.csv)

| Metrica | Friedman χ² | df | p |
|---|---|---|---|
| F1 | 44.0 | 3 | 1.5e-09 |
| BERT base | 483.8 | 2 | 8.6e-106 |
| BERT cons | 563.7 | 2 | 3.9e-123 |

**Lettura:** in ogni metrica gli step *differiscono* in modo molto significativo
anche senza assunzioni parametriche. Le conclusioni dei modelli misti sono
robuste.

---

## 5. Diagnostica iniziale

**File:** [`diagnostics/diagnostics.csv`](diagnostics/diagnostics.csv)

| Metrica | n | ICC(qid) | Shapiro p (normalità residui) |
|---|---|---|---|
| F1 | 14256 | 0.40 | < 1e-43 (non normale) |
| BERT base | 10692 | 0.41 | < 1e-23 (non normale) |
| BERT cons | 10692 | 0.001 | < 1e-47 (non normale) |

**Cosa ci dice:**
- **ICC ~40% su F1/BERTbase**: usare `(1|qid)` era obbligatorio (senza, i
  p-value sarebbero stati artificialmente schiacciati).
- **ICC = 0.001 su BERT cons**: la similarità tra step contigui dipende quasi
  solo dallo step, non dalla domanda. Senso plausibile: la convergenza è una
  proprietà del modello, non del contenuto.
- **Shapiro p ≪ 0.05 ovunque**: i residui non sono normali (specialmente F1, che
  è bimodale 0/1). Per questo F1 è stato modellato con GLMM logistico, e per
  tutto abbiamo confermato i risultati con Friedman (non parametrico).

---

## 6. Limitazioni dei dati attuali

Queste sono le limitazioni note delle analisi presentate qui. Vanno dichiarate
esplicitamente nel paper.

### 6.1 OpenFActScore è incompleto — escluso da questo README

Il file [`rewriting_chains_300q_openfactscore.csv`](../rewriting_chains_300q_openfactscore.csv)
contiene punteggi OFS solo per **55 qid 2-hop** (su 297 totali). Le domande
**3-hop e 4-hop non sono state valutate** con OpenFActScore.

Per non veicolare conclusioni non generalizzabili, **in questo README abbiamo
escluso OFS da tutte le analisi**. I file CSV di OFS rimangono nella cartella
[`inference/`](inference/) (ofs_step_*.csv) ma non sono interpretati qui.

**Conseguenze:**
- **RQ1** ("il rewriting degrada la fattualità?") non è risposta da questo
  documento.
- **RQ2b** ("la complessità influenza il degrado di fattualità?") *non è
  testabile* sulla fattualità: l'OFS ha un solo livello di n-hop. RQ2b qui è
  affrontata solo via Answer F1 e BERTScore (§4.7).

**Cosa serve per chiudere il gap:** rilanciare la pipeline OFS
(`openfactscore_eval.py`, 4-bit su Lisa) sui chains delle **3-hop e 4-hop**.
I dati di rewriting (`rewriting_chains_300q.csv`) sono già completi per tutte
le complessità, quindi serve solo il run di valutazione OFS.

### 6.2 Answer F1 con quantizzazione 4-bit, non bf16

L'Answer F1 attuale è stato calcolato con **OLMo-3.1-32B-Instruct in 4-bit NF4**
(file [`rewriting_chains_300q_answer_f1_olmo31_4bit.csv`](../rewriting_chains_300q_answer_f1_olmo31_4bit.csv)).
Il piano originale prevedeva la valutazione in **bf16 con OLMo-2-32B** su Homer.
I numeri di F1 in 4-bit potrebbero essere sistematicamente diversi da quelli in
bf16 — in particolare la quantizzazione tende a ridurre leggermente le
prestazioni. **I confronti relativi tra step/istruzione/group restano validi**
(la quantizzazione agisce uniformemente sulle condizioni), ma il livello
assoluto di F1 va riportato esplicitando il setup.

### 6.3 Self-Refine non è ancora stato eseguito

**RQ3** ("Self-Refine riduce il degrado?") richiede un esperimento aggiuntivo
con il ciclo di critica/correzione attivo. I dati attuali coprono solo la
pipeline *senza* Self-Refine, quindi RQ3 **resta aperta** ed è un futuro
intervento sperimentale.

### 6.4 Solo 3 step e 3 run

I 3 step di rewriting permettono di osservare l'inizio del trend ma non di
estrapolare l'asintoto (lo step 4 o 5 si stabilizza? continua a calare?). Le
3 run replicate sono sufficienti per concludere che la variabilità tra run è
piccola (ICC < 6%, §4.5), ma non per ottenere CI stretti su singoli qid.

---

## 7. Cosa portare via in due righe

> Il rewriting iterativo **fa perdere capacità di QA quasi tutto al primo step**
> (F1 −17 punti) e produce testi che si stabilizzano rapidamente in un attrattore
> del modello (BERTScore consecutive cresce). content e style si comportano in
> modo qualitativamente simile su F1; differiscono solo nella dinamica di
> convergenza interna. I 3 run sono praticamente intercambiabili. **La lunghezza
> prodotta è un confondente importante**: tutte le istruzioni comprimono
> aggressivamente il testo (mediana 314–967 token contro 2340 dell'originale),
> e una parte del degrado attribuito al rewriting o all'istruzione è in realtà
> un effetto di compressione (§4.8). Le conclusioni sulla fattualità restano
> aperte (OFS incompleto, vedi §6).

---

## 8. Dove trovare cosa

```
results/300q/stats/
├── README.md                  ← questo file (no OFS, vedi §6)
├── diagnostics/
│   └── diagnostics.csv        ← ICC, Shapiro
└── inference/
    ├── inference_summary.csv  ← tutti i p e gli effect size
    ├── inference_summary.md   ← tabella unica
    ├── f1_step_*.csv          ← Test 1 (F1)
    ├── bs_*_step_*.csv        ← Test 2 (BERTScore)
    ├── *_step_x_instr.csv     ← Test 3 (instruction)
    ├── *_step_x_group.csv     ← Test 4 (group)
    ├── *_step_x_hop.csv       ← RQ2b (hop)
    ├── *_degradation_by_hop.csv ← RQ2b
    ├── length_*.csv           ← Test 8 (lunghezza n_tokens)
    ├── run_variance_decomposition.csv ← Test 5 (run)
    ├── corr_*.csv             ← Test 6 (correlazioni)
    ├── robustness_friedman.csv ← Test 9
    └── ofs_*                  ← OFS, NON interpretato qui (incompleto)
```

Per rigenerare tutto:
```
python3.11 scripts/300q/inference_tests.py
```
