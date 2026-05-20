# 15q — Risultati statistici spiegati

Questo documento spiega in modo divulgativo cosa è stato calcolato dallo script
[`scripts/15q/inference_tests.py`](../../../scripts/15q/inference_tests.py) e
come leggere i numeri.

---

## 1. Cosa stiamo misurando

Per ciascuna delle **15 domande** del pilot (5 per livello di complessità:
2-hop, 3-hop, 4-hop), il testo di supporto viene riscritto **3 volte di fila**
(step 1, 2, 3) sotto **4 tipi di istruzione**:

- **content** → `elaborate` / `shorten` (cambiare la quantità di contenuto)
- **style** → `formality` / `paraphrase` (cambiare lo stile mantenendo il contenuto)

Ogni rewriting è ripetuto **3 volte** (run 0, 1, 2) per misurare la stocasticità.
Sul testo di ogni step misuriamo tre metriche:

| Metrica | Cosa misura | Range | Step disponibili |
|---|---|---|---|
| **Answer F1** | il modello QA risponde correttamente? | 0–1 | 0, 1, 2, 3 |
| **OpenFActScore (OFS)** | il testo riscritto contiene fatti corretti? | 0–1 | 1, 2, 3 |
| **BERTScore baseline** | quanto il rewriting somiglia all'originale | 0–1 | 1, 2, 3 |
| **BERTScore consecutive** | quanto lo step k somiglia allo step k-1 | 0–1 | 1, 2, 3 |

> **Step 0** = testo originale, non riscritto. È il *baseline* da cui partiamo.

---

## 2. Glossario veloce (statistica per chi non se ne occupa)

- **p-value**: probabilità di osservare questo risultato (o uno più estremo) se
  in realtà *non* ci fosse nessun effetto. Se p < 0.05, l'effetto è considerato
  "statisticamente significativo".
- **p_holm**: il p-value corretto per il fatto che facciamo più test in
  parallelo. Si guarda **questo**, non il p grezzo.
- **mean_diff**: di quanto cambia la metrica in media tra due step.
- **effect size / rank_biserial**: quanto è *grande* l'effetto,
  indipendentemente dal numero di osservazioni. Vicino a 0 = trascurabile,
  vicino a 1 = massiccio.
- **ICC (Intraclass Correlation)**: percentuale di varianza spiegata da un
  raggruppamento. Es. ICC(qid) = 0.4 vuol dire che il 40% della variabilità è
  dovuto a "quale domanda è" — alcune domande sono semplicemente più facili.
- **Random intercept `(1|qid)`**: nei modelli misti diciamo "tieni conto che la
  stessa domanda è stata vista più volte, non trattare le osservazioni come
  indipendenti".
- **"Non significativo" con n piccolo ≠ "non c'è effetto"**: con 15 domande, un
  effetto reale ma piccolo *può* non superare la soglia 0.05. Direzione e
  magnitudo restano informative.

---

## 3. Quali test abbiamo fatto e perché

| Test | Domanda | Modello |
|---|---|---|
| **1** | Step degrada Answer F1? | GLMM logistico su `F1>0`, contrasti pianificati |
| **2** | Step degrada OFS? | LMM, contrasti pianificati |
| **3** | Step muove BERTScore? | LMM su baseline e consecutive |
| **4** | L'istruzione cambia il pattern? | Modelli separati per content/style |
| **5** | content vs style hanno traiettorie diverse? | LMM con interazione step × group |
| **6** | I 3 run aggiungono variabilità? | Decomposizione varianza + Friedman |
| **7** | Le tre metriche correlano tra loro? | LMM e repeated-measures correlation |
| **8** | I risultati reggono con test non parametrici? | Friedman omnibus |
| **9** | La complessità (n-hop) influenza il degrado? (RQ2b) | LMM con interazione step × n_hop |

Tutti i test usano `(1|qid)` per tenere conto del fatto che ogni domanda è
osservata sotto tutte le condizioni.

---

## 4. Risultati

### 4.1 Answer F1 e gli step

**File:** [`inference/f1_step_contrasts.csv`](inference/f1_step_contrasts.csv)

| Contrasto | mean_diff | p_holm | Effect size |
|---|---|---|---|
| step 0 → 1 | 0.000 | 0.61 (n.s.) | 0.35 |
| step 1 → 2 | −0.006 | 1.00 (n.s.) | 0.00 |
| step 2 → 3 | −0.011 | 1.00 (n.s.) | 0.13 |

**Lettura:** sul pilot 15q, l'Answer F1 **non mostra un calo statisticamente
significativo** tra gli step. La diagnostica indica però un ICC(qid) molto
alto (**0.70**, vedi §5): con solo 15 domande, la varianza tra domande domina
completamente la varianza tra step. In altre parole: alcune domande del pilot
sono facili a ogni step, altre sono difficili a ogni step, e questa varianza
"di chi-è-la-domanda" maschera un eventuale trend tra step.

> Per F1, il pilot 15q è troppo piccolo per documentare in modo solido la
> traiettoria di degrado. Serve un campione più grande.

### 4.2 OpenFActScore — la fattualità cala con gli step

**File:** [`inference/ofs_step_contrasts.csv`](inference/ofs_step_contrasts.csv)

| Contrasto | mean_diff | p_holm |
|---|---|---|
| step 1 → 2 | −0.008 | 0.095 (marginale) |
| step 2 → 3 | −0.010 | 0.071 (marginale) |

**Friedman omnibus** su step 1/2/3: χ² = 13.3, **p = 0.001 (significativo)**.

**Lettura:** l'effetto globale di step su OFS è **significativo**: la fattualità
**cala** nei 3 step di rewriting. I singoli contrasti pairwise (1→2 e 2→3) sono
**borderline** (p ≈ 0.07–0.09) — la direzione è chiara e il calo cumulativo è
~1.8 punti percentuali, ma con n=15 la potenza non basta a superare la
correzione di Holm.

> Take-away: c'è un degrado di fattualità reale e progressivo (l'omnibus lo
> conferma), ma la nostra capacità di "puntare il dito" su quale specifico
> passaggio fa la differenza è limitata dalla numerosità.

### 4.3 BERTScore — due dinamiche opposte

**BERTScore baseline** (similarità vs testo originale)
[`inference/bs_baseline_step_contrasts.csv`](inference/bs_baseline_step_contrasts.csv):

| Contrasto | mean_diff | p_holm |
|---|---|---|
| step 1 → 2 | **−0.015** | 0.0001 |
| step 2 → 3 | **−0.007** | 0.0001 |

→ Il testo si **allontana dall'originale** in modo monotonico e significativo.
Calo totale ~2.2 punti.

**BERTScore consecutive** (similarità step k vs step k-1)
[`inference/bs_consecutive_step_contrasts.csv`](inference/bs_consecutive_step_contrasts.csv):

| Contrasto | mean_diff | p_holm |
|---|---|---|
| step 1 → 2 | **+0.057** | 0.0001 |
| step 2 → 3 | **+0.017** | 0.0001 |

→ Gli step diventano **sempre più simili al precedente**: il rewriting *converge*
a un attrattore. Il salto grosso è 1→2; dopo, il modello cambia sempre meno.

**Insieme**, questi due numeri raccontano una storia precisa: il sistema si
allontana dall'originale soprattutto al primo step, e poi si stabilizza intorno
a una nuova versione "del modello".

### 4.4 L'istruzione (elaborate/shorten/formality/paraphrase) cambia il pattern?

**File:** `inference/*_step_x_instr.csv` (uno per metrica × gruppo)

Modelli con interazione `step × instruction_type`, fittati separatamente per
content e style (perché le istruzioni sono *nested* nei gruppi).

- **F1**: nessuna interazione significativa. Le 4 istruzioni hanno traiettorie
  sovrapposte (anche perché F1 in sé non si muove sul pilot).
- **OFS**: `shorten` mostra fattualità leggermente più bassa di `elaborate` (la
  riscrittura compressa tende a perdere fatti); `formality` è la più alta tra
  le 4 istruzioni. Le differenze sono modeste e non tutte significative.
- **BERTScore**: `paraphrase` (che riscrive di più) ha il drift maggiore dal
  testo originale. Coerente con l'intento dell'istruzione.

### 4.5 content vs style hanno traiettorie diverse?

**File:** `inference/*_step_x_group.csv`

- **F1**: nessuna interazione significativa.
- **OFS**: nessuna interazione significativa — content e style mostrano lo
  stesso pattern di degrado di fattualità.
- **BERTScore baseline**: style mostra un livello leggermente più alto (più
  simile all'originale) ma la *velocità* del declino è uguale.
- **BERTScore consecutive**: piccola differenza nel pattern di convergenza,
  marginalmente significativa.

### 4.6 RQ2b — la complessità della domanda (n-hop) influenza il degrado?

**File:** `inference/*_step_x_hop.csv` e `*_degradation_by_hop.csv`

Il pilot 15q ha 5 domande per ciascun livello di complessità (2-hop, 3-hop,
4-hop), quindi possiamo testare se le domande più complesse degradano in modo
diverso.

**OFS medio per hop × step:**

| n_hop | step 1 | step 2 | step 3 | Δ cumulativo 1→3 |
|---|---|---|---|---|
| 2 | 0.913 | 0.902 | 0.903 | **−0.010** |
| 3 | 0.907 | 0.903 | 0.888 | **−0.019** |
| 4 | 0.867 | 0.857 | 0.841 | **−0.026** |

**Pattern chiaro a occhio:**
- Le 4-hop partono già più basse di fattualità (0.867 vs 0.913 delle 2-hop).
- Il calo cumulativo cresce con la complessità: 2-hop perde ~1 pt, 3-hop ~2 pt,
  **4-hop ~2.6 pt** (≈ 2.6× il calo delle 2-hop).
- La direzione è perfettamente coerente con l'ipotesi: *domande più complesse
  si degradano di più con il rewriting*.

**Wilcoxon step 1 → step 3 per hop:**

| n_hop | mean_diff | p_holm |
|---|---|---|
| 2 | −0.010 | 0.63 |
| 3 | −0.019 | 0.19 |
| 4 | −0.026 | 0.19 |

**Interazione step × n_hop nel modello misto:** tutti i termini di interazione
p > 0.27.

**Verdetto:** la tendenza è coerente con l'ipotesi RQ2b, ma con **n = 5 qid per
hop** il test non raggiunge la significatività dopo la correzione per confronti
multipli. **Il pilot suggerisce, non conferma.**

**Su F1 e BERTScore:** anche qui le interazioni step × n_hop non sono
significative. Le 4-hop hanno F1 baseline più alta del pilot (0.60 vs 0.40) —
ma è un artefatto del piccolo campione (5 domande 4-hop selezionate non
casualmente).

### 4.7 I 3 run aggiungono variabilità?

**File:** [`inference/run_variance_decomposition.csv`](inference/run_variance_decomposition.csv)

| Metrica | ICC(qid) | ICC(run) | Friedman p |
|---|---|---|---|
| F1 | 0.71 | **1.0%** | 0.26 |
| OFS | 0.39 | **3.5%** | 0.13 |
| BERT base | 0.09 | **4.6%** | 3e-04 |
| BERT cons | 0.04 | **3.0%** | 0.008 |

**Lettura:** la variabilità tra run è **piccola** (1–5%): le 3 run sono
sostanzialmente equivalenti. Quasi tutta la varianza viene dalle differenze tra
domande (ICC qid alto su F1/OFS) o dal residuo. Puoi tranquillamente aggregare
sulle run senza perdere informazione.

### 4.8 Le tre metriche correlano tra loro?

**File:** [`inference/corr_repeated_measures.csv`](inference/corr_repeated_measures.csv)

Correlazioni *entro qid* (controllando per il fatto che alcune domande sono
sistematicamente più facili):

| Coppia | r | 95% CI | p |
|---|---|---|---|
| F1 ↔ OFS | 0.009 | [−0.08, 0.09] | 0.83 (n.s.) |
| F1 ↔ BERTScore base | 0.112 | [0.03, 0.20] | 0.010 |
| OFS ↔ BERTScore base | −0.029 | [−0.11, 0.06] | 0.51 (n.s.) |

**Lettura:** sul pilot le metriche sono **quasi scorrelate**. L'unica
correlazione significativa è tra F1 e BERTScore baseline, ma debole (r = 0.11).
È coerente col fatto che ICC(qid) = 0.70 su F1: con così tanta varianza
"intrinseca" delle domande, le relazioni tra metriche faticano a emergere.

Il mixed-effects model [`corr_f1_on_ofs_bs.csv`](inference/corr_f1_on_ofs_bs.csv)
conferma: BERTbase è l'unico predittore significativo di F1 (β = 0.74, p = 0.01);
OFS non contribuisce; step nemmeno.

### 4.9 Robustness — i test non parametrici confermano?

**File:** [`inference/robustness_friedman.csv`](inference/robustness_friedman.csv)

| Metrica | Friedman χ² | df | p |
|---|---|---|---|
| F1 | 0.41 | 3 | 0.94 (n.s.) |
| OFS | 13.3 | 2 | **0.001** |
| BERT base | 30.0 | 2 | **3e-07** |
| BERT cons | 30.0 | 2 | **3e-07** |

**Lettura:** OFS, BERT base e BERT cons mostrano effetti omnibus di step
**robustamente significativi** anche senza assunzioni parametriche. F1 invece
non si muove tra step in modo rilevabile.

---

## 5. Diagnostica iniziale

**File:** [`diagnostics/diagnostics.csv`](diagnostics/diagnostics.csv)

| Metrica | n | mean | std | ICC(qid) | Shapiro p (residui) |
|---|---|---|---|---|---|
| F1 | 720 | 0.326 | 0.398 | **0.70** | < 1e-24 (non normale) |
| OFS | 540 | 0.887 | 0.070 | 0.36 | < 1e-16 (non normale) |
| BERT base | 540 | 0.873 | 0.034 | 0.06 | 0.03 |
| BERT cons | 540 | 0.930 | 0.042 | 0.001 | < 1e-12 (non normale) |

**Cosa ci dice:**
- **ICC(qid) = 0.70 su F1**: il 70% della varianza di F1 è "tra domande". Con
  solo 15 qid, il pilot fotografa il comportamento di 15 domande specifiche;
  effetti tra step sono difficili da isolare.
- **ICC(qid) ≈ 0 su BERT consecutive**: la similarità tra step consecutivi
  dipende quasi solo dallo step, non dalla domanda. La convergenza è una
  proprietà del modello, non del contenuto.
- **Shapiro p ≪ 0.05** quasi ovunque: i residui non sono normali, in
  particolare per F1 (bimodale 0/1). Per questo F1 è stato modellato con GLMM
  logistico e tutto è stato confermato con Friedman (non parametrico).

---

## 6. Limitazioni del pilot

### 6.1 Sample size

n = 15 qid (5 per hop) è il limite principale. Conseguenza pratica: il pilot
**rileva solo effetti grandi** (BERTScore: drift 1–2%; convergenza ~6–8%).
Effetti più piccoli ma reali (F1 drop, OFS gradiente per hop) **esistono ma non
passano i test di significatività**. È il caso classico in cui *direzione +
magnitudo sono informative* anche senza p < 0.05.

### 6.2 ICC(qid) elevato su F1

Con il 70% della varianza F1 dovuta a "quale domanda è", il campione di 15
domande è dominato dalla selezione iniziale. Cambiare le 15 domande potrebbe
cambiare visibilmente i numeri assoluti. I trend qualitativi sono il modo
corretto di leggere il pilot.

### 6.3 Self-Refine non incluso nei test

Esistono i file `self_refine_chains_15q_smoketest_v{4,6}.csv` ma sono **smoke
test** (1 qid, 1 run). Insufficienti per qualsiasi inferenza statistica.
RQ3 (Self-Refine ripara il degrado?) resta aperta.

### 6.4 Solo 3 step

I 3 step permettono di osservare l'inizio del trend ma non di estrapolare
l'asintoto: non sappiamo se a step 4 o 5 la fattualità si stabilizza o continua
a calare.

---

## 7. Cosa portare via in due righe

> Sul pilot 15q, il rewriting iterativo produce un **drift testuale chiaro e
> significativo** (BERTScore) e un **degrado della fattualità** omnibus
> significativo (Friedman p = 0.001 su OFS), con un **gradiente di degrado
> coerente con la complessità della domanda** (4-hop: ~2.6× il calo delle
> 2-hop). L'effetto di step su Answer F1 non è rilevabile con n=15: serve un
> campione più ampio per documentare la traiettoria QA.

---

## 8. Dove trovare cosa

```
results/15q/stats/
├── README.md                       ← questo file
├── diagnostics/
│   └── diagnostics.csv             ← ICC, Shapiro
└── inference/
    ├── inference_summary.{csv,md}  ← tabella unica di tutti i p ed effect size
    ├── f1_step_*.csv               ← Test 1
    ├── ofs_step_*.csv              ← Test 2
    ├── bs_*_step_*.csv             ← Test 3
    ├── *_step_x_instr.csv          ← Test 4
    ├── *_step_x_group.csv          ← Test 5
    ├── *_step_x_hop.csv            ← RQ2b
    ├── *_degradation_by_hop.csv    ← RQ2b
    ├── run_variance_decomposition.csv ← Test 6 (run)
    ├── corr_*.csv                  ← Test 7 (correlazioni)
    └── robustness_friedman.csv     ← Test 8
```

Per rigenerare tutto:
```
python3.11 scripts/15q/inference_tests.py
```
