# Riepilogo dei risultati — Rewriting Pipeline su MuSiQue (OLMo-3.1-32B-Instruct)

Questo documento riassume in un unico posto i risultati dei README statistici del progetto:

- [`results/15q/stats/README.md`](15q/stats/README.md) — pilot 15 domande
- [`results/300q/stats/README.md`](300q/stats/README.md) — scala completa 300 domande
- [`results/300q/stats/elaborate/README.md`](300q/stats/elaborate/README.md) — analisi dedicata a `elaborate`
- [`results/300q/stats/recovery/README.md`](300q/stats/recovery/README.md) — analisi del recupero F1

Per i CSV e i modelli completi fare riferimento ai README specifici.

---

## Setup sperimentale

Per ogni domanda di MuSiQue, il testo di supporto viene riscritto **3 volte di fila** (step 1, 2, 3)
sotto **4 tipi di istruzione**:

- **content** → `elaborate` (arricchisci il testo) / `shorten` (accorcialo)
- **style** → `formality` (rendi più formale) / `paraphrase` (parafrasa mantenendo il contenuto)

Ogni rewriting è ripetuto **3 volte indipendenti** (run 0, 1, 2) per misurare
quanto i risultati variano per effetto della stocasticità del modello.

Le metriche calcolate su ogni step sono:

| Metrica | Cosa misura | Range | Step disponibili |
|---|---|---|---|
| **Answer F1** | il modello QA risponde correttamente al testo riscritto? | 0–1 | 0, 1, 2, 3 |
| **OpenFActScore (OFS)** | quante affermazioni nel testo sono supportate da una fonte? | 0–1 | 1, 2, 3 |
| **BERTScore baseline** | quanto il testo riscritto somiglia semanticamente all'originale (step 0)? | 0–1 | 1, 2, 3 |
| **BERTScore consecutive** | quanto lo step k somiglia semanticamente allo step k−1? | 0–1 | 1, 2, 3 |

> **Step 0** = testo originale, non riscritto. È il punto di partenza.

**Grafico di riferimento per le 300q:** [`plots/300q/`](plots/300q/)

---

## 1. Pilot 15q

**Dimensione:** 15 domande (5 per complessità: 2-hop, 3-hop, 4-hop), 4 istruzioni, 3 run = 720 osservazioni totali per metrica.

### Come abbiamo testato

Per tutte le metriche abbiamo usato **modelli misti lineari o logistici (LMM/GLMM)** con random intercept `(1|qid)`. Questo è necessario perché ogni domanda viene osservata in tutte le condizioni: senza questo accorgimento, le osservazioni sembrerebbero indipendenti e i p-value risulterebbero artificialmente piccoli. I contrasti tra step sono stati pianificati a priori (step0→1, step1→2, step2→3) e corretti per confronti multipli con il metodo di Holm. I test sono stati confermati con il Friedman test non parametrico, che non fa assunzioni sulla distribuzione dei dati.

### Answer F1 — nessun calo rilevabile sul pilot

Sul pilot, l'F1 non mostra un calo statisticamente significativo tra gli step
(contrasto step0→1: Δ = 0.000, p_holm = 0.61; Friedman omnibus p = 0.94).
Questo non significa che l'effetto non esista: l'**ICC(qid) = 0.70** indica
che il 70% della varianza di F1 è spiegato da "quale domanda è" — alcune
domande sono facili a ogni step, altre difficili a ogni step. Con solo 15 domande,
questa varianza tra domande domina completamente la varianza tra step,
rendendo impossibile rilevare trend moderati. Il pilot era troppo piccolo per
questo effetto; il 300q lo documenterà.

### OpenFActScore — degrado significativo, ma granularità limitata

| Step | OFS medio |
|---|---|
| 1 | 0.895 |
| 2 | 0.887 |
| 3 | 0.877 |

Il Friedman omnibus su step 1/2/3 è significativo (**χ² = 13.3, p = 0.001**):
la fattualità cala con il rewriting iterativo. I singoli contrasti pairwise
(step1→2: Δ = −0.008, p_holm = 0.095; step2→3: Δ = −0.010, p_holm = 0.071)
sono borderline — la direzione è chiara e il calo cumulativo è ~1.8 punti
percentuali, ma con n=15 la potenza non basta a isolare quale specifico
passaggio fa la differenza.

### BERTScore — due dinamiche chiare

**BERTScore baseline** (confronto con il testo originale, step 1–3: 0.886 → 0.871 → 0.863):
il testo si allontana dall'originale in modo monotonico e altamente significativo
(step1→2: Δ = −0.016, p_holm < 0.001; step2→3: Δ = −0.007, p_holm < 0.001).

**BERTScore consecutive** (confronto tra step contigui, step 1–3: ~0.887 → 0.944 → 0.961):
ogni step diventa più simile al precedente (step1→2: Δ = +0.057, p_holm < 0.001;
step2→3: Δ = +0.017, p_holm < 0.001). Il rewriting **converge a un attrattore**:
il salto maggiore è tra step 1 e 2, poi il modello cambia sempre meno.

Letti insieme: il sistema si allontana dall'originale soprattutto al primo step,
e si stabilizza rapidamente attorno a una versione "propria del modello".

### RQ2b — complessità e degrado

Sul pilot il pattern è coerente con l'ipotesi — le 4-hop degradano di più:

| n_hop | OFS step 1 | OFS step 3 | Δ cumulativo |
|---|---|---|---|
| 2-hop | 0.913 | 0.903 | −0.010 |
| 3-hop | 0.907 | 0.888 | −0.019 |
| 4-hop | 0.867 | 0.841 | **−0.026** (~2.6× le 2-hop) |

Le 4-hop partono già da una fattualità più bassa e peggiorano di più. Tuttavia
con n=5 qid per livello di hop i test non raggiungono la significatività dopo
correzione (p_holm > 0.19 per tutti i contrasti). **Il pilot suggerisce, non
conferma.**

### Variabilità tra run

La variabilità tra run è trascurabile su tutte le metriche (ICC run: 1–5%
vs ICC qid: 4–71%). Si può aggregare sulle run senza perdere informazione.

### Conclusione 15q

> Il pilot documenta un **drift testuale reale e significativo** (BERTScore)
> e un **degrado di fattualità globalmente significativo** (OFS, Friedman
> p = 0.001), con un gradiente coerente con la complessità della domanda.
> L'effetto su Answer F1 non è rilevabile con n=15.

---

## 2. Scala completa 300q

**Dimensione:** 297 domande (100 2-hop, ~97 3-hop, 100 4-hop), 4 istruzioni, 3 run.

**Nota OFS:** sul run 300q l'OpenFActScore è stato calcolato solo sulle
55 domande 2-hop (su 297 totali). I risultati OFS per il 300q sono trattati
**separatamente in fondo a questa sezione (§2.8)** per non mescolarli con le
metriche complete.

### Come abbiamo testato

La struttura è identica al pilot, ma con molte più osservazioni (14256 per F1,
10692 per BERTScore). Il modello di riferimento è:

- **F1**: GLMM logistico su F1>0 (la distribuzione è bimodale: molti 0 e 1),
  con `(1|qid)`. Contrasti pianificati step-by-step, corretti con Holm.
- **BERTScore**: LMM lineare con `(1|qid)`.
- **Robustness**: tutti i risultati confermati con Friedman test non parametrico.

Il Friedman omnibus su step è significativo per tutte le metriche
(F1: **χ² = 44.0, p = 1.5×10⁻⁹**; BERTScore base: **χ² = 483.8, p = 8.6×10⁻¹⁰⁶**;
BERTScore cons: **χ² = 563.7, p = 3.9×10⁻¹²³**), confermando che le
distribuzioni differiscono tra gli step indipendentemente da assunzioni parametriche.

---

### 2.1 Answer F1 — il crollo avviene al primo step

Medie per step (il grafico di riferimento è [`plots/300q/png/answer_f1_by_step_olmo31_4bit.png`](plots/300q/png/answer_f1_by_step_olmo31_4bit.png)):

| Step | F1 medio |
|---|---|
| **0** (testo originale) | **0.362** |
| 1 | 0.215 |
| 2 | 0.191 |
| 3 | 0.177 |

Contrasti pianificati (Wilcoxon paired, corretti Holm):

| Contrasto | Δ medio | Effect size (rank-biserial) | p_holm |
|---|---|---|---|
| step 0 → 1 | **−0.169** | **0.377 (medio)** | **1.1×10⁻⁸** |
| step 1 → 2 | −0.030 | 0.019 | 0.93 (n.s.) |
| step 2 → 3 | −0.017 | 0.044 | 0.93 (n.s.) |

Il GLMM logistico sui coefficienti di step (rispetto a step 0 come baseline)
conferma la stessa storia: step1 = −0.169 (p = 7.8×10⁻⁹³), step2 = −0.198
(p = 1.0×10⁻¹²⁷), step3 = −0.216 (p = 2.4×10⁻¹⁵⁰). L'effetto si accumula
ma il salto principale è già avvenuto al primo step.

**Lettura:** già una sola riscrittura fa quasi tutto il danno — l'F1 crolla di
~17 punti percentuali. Gli step successivi aggiungono un ulteriore calo che,
pur presente in modo assoluto, non è statisticamente distinguibile da zero
rispetto al crollo del primo step (p_holm = 0.93 per entrambi i contrasti).

---

### 2.2 Answer F1 per complessità (n-hop)

Medie per hop × step (grafico: [`plots/300q/png/answer_f1_by_hop_olmo31_4bit.png`](plots/300q/png/answer_f1_by_hop_olmo31_4bit.png)):

| n_hop | step 0 | step 1 | step 2 | step 3 | Δ cumulativo (0→3) |
|---|---|---|---|---|---|
| **2-hop** | 0.407 | 0.241 | 0.204 | 0.188 | −0.219 |
| **3-hop** | 0.377 | 0.231 | 0.211 | 0.193 | −0.184 |
| **4-hop** | 0.303 | 0.175 | 0.157 | 0.150 | −0.153 |

I dati mostrano due pattern distinti. Primo, le **4-hop partono già più
basse** sul testo originale (F1 = 0.303 vs 0.407 delle 2-hop): domande più
complesse sono più difficili da rispondere indipendentemente dal rewriting.
Secondo, il calo cumulativo in valore assoluto è leggermente minore per 4-hop
(−0.153) rispetto a 2/3-hop (−0.184/−0.219), ma le 4-hop partendo più basse
perdono proporzionalmente una quota simile.

Il modello GLMM con interazione `step × n_hop` mostra che le 4-hop hanno una
traiettoria di degrado significativamente diversa a step 2 e 3 rispetto alle
2-hop (interazione step2 × 4-hop: β = +0.041, p = 0.042; step3 × 4-hop:
β = +0.052, p = 0.010). Il test Wilcoxon tra il minimo e il massimo F1 per
hop è significativo per tutti i livelli (2-hop: p_holm = 0.00015; 3-hop:
p_holm = 0.00015; 4-hop: p_holm = 0.008).

---

### 2.3 Answer F1 per istruzione

Medie per istruzione × step (grafico: [`plots/300q/png/answer_f1_by_instruction_olmo31_4bit.png`](plots/300q/png/answer_f1_by_instruction_olmo31_4bit.png)):

| Istruzione | step 0 | step 1 | step 2 | step 3 |
|---|---|---|---|---|
| elaborate | 0.362 | 0.209 | 0.187 | 0.170 |
| formality | 0.362 | 0.233 | 0.200 | 0.186 |
| paraphrase | 0.362 | 0.213 | 0.193 | 0.184 |
| shorten | 0.362 | 0.207 | 0.182 | 0.168 |

Il GLMM con interazione `step × instruction_type`, fittato separatamente per
content (elaborate vs shorten) e per style (formality vs paraphrase), non
mostra interazioni significative in nessuno dei due gruppi (tutti i p_interazione
> 0.53). Le traiettorie sono sostanzialmente sovrapposte. In apparenza `formality`
sembra la migliore, ma questa differenza è largamente spiegata dalla lunghezza
prodotta (vedi §2.6 — mediation). **A parità di lunghezza, l'istruzione conta poco per F1.**

---

### 2.4 BERTScore — drift e convergenza

Grafico di riferimento: [`plots/300q/png/bertscore_baseline_by_step.png`](plots/300q/png/bertscore_baseline_by_step.png) e [`plots/300q/png/bertscore_consecutive_by_step.png`](plots/300q/png/bertscore_consecutive_by_step.png).

**BERTScore baseline** (quanto ogni step somiglia al testo originale):

| Step | Medio | Step 1→2 Δ | Step 2→3 Δ |
|---|---|---|---|
| 1 | 0.853 | — | — |
| 2 | 0.845 | **−0.008** (p_holm = 5.1×10⁻⁴⁹) | — |
| 3 | 0.841 | — | **−0.004** (p_holm = 1.5×10⁻⁴⁶) |

Il testo si allontana dall'originale in modo monotonico e altamente significativo,
con effect size quasi massicci (rank-biserial: 0.989 e 0.959). Il calo
assoluto è però contenuto (~1 punto percentuale totale): il testo cambia in
modo statisticamente rilevabile ma non drasticamente.

**BERTScore consecutivo** (quanto ogni step somiglia al precedente):

| Step | Medio | Step 1→2 Δ | Step 2→3 Δ |
|---|---|---|---|
| 1 | ~0.857 | — | — |
| 2 | ~0.936 | **+0.085** (p_holm = 3.8×10⁻⁵⁰) | — |
| 3 | ~0.951 | — | **+0.015** (p_holm = 1.5×10⁻⁴⁷) |

Il salto positivo da step 1 a step 2 (+0.085) è il più grande: vuol dire che
il testo di step 2 somiglia molto di più a quello di step 1 di quanto step 1
somigliasse all'originale. Il rewriting converge rapidamente a un attrattore —
una versione "propria del modello" da cui tende a riprodurre variazioni simili.

Questi due andamenti raccontano la stessa storia da angolature diverse: il
sistema si allontana dall'originale soprattutto al primo step, e poi si
stabilizza.

---

### 2.5 BERTScore per complessità e per gruppo

**Per hop** (grafico: [`plots/300q/png/bertscore_baseline_by_hop.png`](plots/300q/png/bertscore_baseline_by_hop.png)):

| n_hop | BS_baseline step 1 | step 2 | step 3 |
|---|---|---|---|
| 2-hop | 0.857 | 0.847 | 0.843 |
| 3-hop | 0.855 | 0.847 | 0.842 |
| 4-hop | 0.849 | 0.841 | 0.837 |

Le 4-hop hanno un BERTScore baseline sistematicamente più basso (β = −0.008,
p = 0.040 nel modello misto): il rewriting si allontana leggermente di più
dall'originale per domande più complesse. La velocità del drift è però la
stessa per tutti i livelli di hop (nessuna interazione step × hop significativa,
tutti i p > 0.13).

Sul BERTScore consecutivo, le **4-hop convergono all'attrattore più velocemente**
(interazione step2 × 4-hop: β = +0.012, p = 7.7×10⁻¹⁰; step3 × 4-hop:
β = +0.008, p = 1.4×10⁻⁵): per testi più complessi il modello si assesta in
una nuova versione prima.

**Per gruppo** (content vs style):

| Gruppo | BS_baseline step 1 | step 2 | step 3 |
|---|---|---|---|
| content | 0.844 | 0.836 | 0.832 |
| style | 0.863 | 0.854 | 0.849 |

Style è sistematicamente ~2 pp più simile all'originale (β = +0.019, p = 2.4×10⁻⁸⁶):
le istruzioni di stile (formality, paraphrase) mantengono più lessico dell'originale
rispetto a quelle di contenuto (elaborate, shorten). La velocità del declino è
però uguale (nessuna interazione step × group significativa su BS_baseline,
p ≥ 0.11).

Sul BERTScore consecutivo, **content converge all'attrattore più velocemente di
style** (interazione step2 × style: β = −0.012, p = 3.5×10⁻¹⁴; step3 × style:
β = −0.013, p = 1.1×10⁻¹⁶): le riscritture di contenuto si stabilizzano prima
su una versione ricorrente; quelle di stile continuano a variare un po' di più.

---

### 2.6 La lunghezza prodotta è un confondente importante ★

Tutte le istruzioni **comprimono** il testo originale (mediana originale: ~2340 token),
con percentuali di catene "catastrofiche" (<200 token) non trascurabili:

| Istruzione | step 1 (media) | step 2 | step 3 | % catene <200 tok a step 1 |
|---|---|---|---|---|
| elaborate | 710 | 647 | 620 | **32.1%** |
| formality | 935 | 722 | 631 | 10.1% |
| paraphrase | 555 | 420 | 357 | **29.7%** |
| shorten | 458 | 352 | 308 | 17.7% |

La lunghezza predice F1 in modo robusto. Il modello `F1 ~ log(n_tokens) + step + (1|qid)` mostra:

| Termine | Coef | p |
|---|---|---|
| log(n_tokens) | **+0.038** | <0.001 |
| step 1 | −0.085 | <0.001 |
| step 2 | −0.104 | <0.001 |
| step 3 | −0.114 | <0.001 |

Per ogni raddoppio della lunghezza prodotta, l'F1 cresce di ~0.026 punti.
Più informazione nel testo → più probabilità di trovare la risposta.

**Mediation — l'effetto dell'istruzione su F1 cambia quando si controlla per n_tokens:**

Senza controllare per la lunghezza, `formality` appare la migliore (+0.017 F1
rispetto a `elaborate`, p = 0.016) e `shorten` la peggiore (−0.003, n.s.).
Aggiungendo `log(n_tokens)` al modello, il quadro si inverte:

| Istruzione vs elaborate | Senza n_tokens | Con n_tokens |
|---|---|---|
| formality | +0.017 (p = 0.016) | +0.003 (p = 0.71, n.s.) |
| paraphrase | +0.008 (p = 0.28, n.s.) | **+0.026 (p < 0.001)** |
| shorten | −0.003 (p = 0.63, n.s.) | **+0.018 (p = 0.016)** |
| log(n_tokens) | — | **+0.047 (p < 0.001)** |

La superiorità apparente di `formality` era interamente dovuta al fatto che
produce testi più lunghi. Controllando per la lunghezza, `paraphrase` e
`shorten` emergono come le istruzioni che producono testi più utili per QA
a parità di token — cioè che conservano meglio le informazioni rilevanti.
`elaborate` invece comprime in modo non selettivo (vedi §3).

Analogamente, l'effetto di step su F1 si riduce di ~30% controllando per
n_tokens (da −0.114 a −0.080 a step 3): parte di quello che attribuivamo
al rewriting iterativo è in realtà compressione progressiva del testo.

---

### 2.7 Variabilità tra run e correlazione tra metriche

**Run:** la variabilità tra le 3 run è piccola su tutte le metriche (ICC run:
3.1% su F1, 5.6% su BS_base, 2.2% su BS_cons), contro un ICC(qid) di ~40–43%
su F1 e BS_base. Quasi tutta la varianza è spiegata da "quale domanda è" e
dal residuo, non da "quale run". Le 3 run sono praticamente intercambiabili.

**Correlazioni tra metriche** (repeated-measures correlation, entro qid):

| Coppia | r | p |
|---|---|---|
| F1 ↔ BERTScore baseline | 0.193 | 2.3×10⁻¹⁷ |
| F1 ↔ OFS (su 55 2-hop) | 0.094 | 4.3×10⁻⁵ |
| OFS ↔ BERTScore baseline | 0.252 | 7.1×10⁻²⁹ |

Le correlazioni esistono ma sono deboli. La similarità lessicale all'originale
(BERTScore) è solo un proxy approssimativo della capacità di rispondere
correttamente (F1): i due costrutti misurano cose diverse.

---

### 2.8 OpenFActScore su 300q — nota separata (solo 55 qid 2-hop)

> **Attenzione alla generalizzabilità.** Sul run 300q l'OFS è stato calcolato
> solo sulle **55 domande 2-hop** (su 297 totali). Le 3-hop e le 4-hop non sono
> state valutate su questa metrica. I risultati qui sotto sono quindi
> **specifici delle domande più semplici del dataset** e non rappresentativi
> dell'intero esperimento.

Medie OFS per step (solo 2-hop, n=55):

| Step | OFS medio |
|---|---|
| 1 | 0.881 |
| 2 | 0.865 |
| 3 | 0.852 |

Il calo è progressivo e significativo: step1→2: Δ = −0.017 (p = 6.1×10⁻⁷);
step2→3: Δ = −0.013 (p = 2.2×10⁻⁵). Friedman omnibus: **χ² = 52.6, p = 3.7×10⁻¹²**.
Ogni rewriting introduce nuovi errori fattuali, indipendentemente dall'istruzione.

Sulla **recall dei fatti** (analisi su un sottoinsieme dell'istruzione `elaborate`):
la proporzione di affermazioni dell'originale che sopravvivono nel testo riscritto
cala da 0.586 a step 1, a 0.490 a step 2, fino a 0.416 a step 3. Già dopo la
prima riscrittura il 41% delle informazioni è andata perduta; al terzo step,
quasi il 60%. Questo suggerisce che la caduta di F1 non è solo dovuta
all'introduzione di errori, ma in larga parte alla **perdita di informazione**
(i fatti rilevanti per rispondere semplicemente non compaiono più nel testo).

Per avere OFS completo su tutti i livelli di hop bisogna rilanciare la pipeline
OFS sulle 3-hop e 4-hop (i dati di rewriting sono già disponibili).

---

## 3. `elaborate` non elabora — analisi dedicata (300q)

L'analisi dettagliata sull'istruzione `elaborate` su 891 catene (300q) rivela
un comportamento controintuitivo.

**Il modello non elabora mai.** Su nessuna delle 891 catene il testo prodotto
è più lungo dell'originale (0.0% a ogni step). Di fatto `elaborate` si comporta
come una compressione blanda:

| Step | Lunghezza media | % catene che si dimezzano | % catene <200 token |
|---|---|---|---|
| 0 (originale) | 2445 tok | — | — |
| 1 | 710 tok | 73.3% | **32.1%** |
| 2 | 647 tok | 83.6% | 27.6% |
| 3 | 620 tok | 87.0% | 21.4% |

Il confronto con `shorten` mostra che le due istruzioni producono lunghezze
significativamente diverse (elaborate ~252 tok in più a step 1, p < 0.001),
quindi non sono identiche — ma `elaborate` non aumenta mai la lunghezza
dell'originale. Si comporta come "shorten leggera".

**Effetto del collasso sulle metriche.** Le catene che collassano sotto i 200
token (catene "corte") hanno:

| Metrica | Catene corte (<200 tok) | Catene lunghe (≥200 tok) | Δ | p_holm |
|---|---|---|---|---|
| **F1** | 0.117 | 0.216 | **−0.099** | <0.001 |
| **BERTScore baseline** | 0.801 | 0.853 | **−0.052** | <0.001 |
| **OFS** | 0.865 | 0.872 | −0.007 | 0.033 |

L'F1 è quasi dimezzato per le catene corte. L'OFS invece resta sorprendentemente
stabile (differenza di appena 0.7 pp): il poco testo che resta è ancora
fattualmente corretto, ma è così scarno da non contenere le informazioni
necessarie per rispondere alla domanda.

**Le 4-hop sono le più vulnerabili al collasso:**

| n_hop | % catene corte step 1 |
|---|---|
| 2-hop | 29% |
| 3-hop | 27% |
| 4-hop | **40%** |

I testi originali delle 4-hop sono più lunghi e complessi, e il modello
sembra trattarli in modo più aggressivo.

**Mediation su OFS.** Il modello `OFS ~ instruction_type + step + (1|qid)`
senza n_tokens mostra differenze significative tra istruzioni (es. formality
vs elaborate: β = +0.014, p = 0.04). Aggiungendo `log(n_tokens)`, tutti i
coefficienti di instruction_type collassano a zero (log_ntokens: β = +0.060,
p < 0.001; formality vs elaborate: β = −0.003, p = 0.66). La differenza di
fattualità tra istruzioni è **quasi interamente spiegata dalla lunghezza
prodotta**, non dall'istruzione in sé. Due testi della stessa lunghezza hanno
OFS simile indipendentemente da quale istruzione li ha generati.

> **Nota di raccordo con §4 (recovery).** I risultati appena descritti —
> elaborate comprime, F1 peggiora, le catene corte sono quasi inutili — sembrano
> contraddire il fatto che nella sezione successiva `elaborate` risulti
> l'istruzione con il **tasso di recupero più alto** (25.1%). La contraddizione
> è solo apparente: i due fenomeni riguardano **popolazioni di partenza diverse**.
>
> L'analisi di §3 descrive cosa succede alle chain che **partono con F1 > 0**
> sul testo originale: lì `elaborate` danneggia, perché comprime un testo che
> già funzionava, facendo perdere informazione. L'analisi di recovery in §4
> descrive invece cosa succede alle chain che **partono con F1 = 0** — cioè
> casi in cui il testo originale era già insufficiente per rispondere. In quel
> contesto, anche una riscrittura compressa può riorganizzare e riformulare il
> contenuto in modo più navigabile per il modello QA, risultando migliore del
> punto di partenza. `elaborate` è quindi la peggiore istruzione quando si
> parte da un testo che funziona, e la migliore quando il testo di partenza
> già non funziona.

---

## 4. Recovery — il rewriting può anche "riparare" (300q)

Analisi su 3564 chain totali (297 qid × 4 istruzioni × 3 run).

**Il 50.5% delle chain parte con F1=0 sul testo originale** (1800 su 3564).
Di queste, **394 (21.9%) vengono recuperate** — cioè raggiungono F1 > 0
in almeno uno step successivo.

### Chi recupera meglio?

**Per istruzione** (Cochran's Q omnibus: **Q = 10.15, p = 0.017** → l'effetto
dell'istruzione è significativo):

| Istruzione | % recovered |
|---|---|
| **elaborate** | **25.1%** |
| paraphrase | 22.2% |
| shorten | 20.9% |
| formality | 19.3% |

Nel GLMM logistico `recovered ~ instruction_type + (1|qid)`, `formality` è
significativamente peggiore di `elaborate` (β = −0.058, p = 0.008); `shorten`
borderline (β = −0.042, p = 0.054). La differenza elaborare vs formality è
di ~6 punti percentuali in termini di tasso di recupero.

**Per complessità** (chi² omnibus: **χ² = 21.6, p = 2×10⁻⁵**; trend di
Cochran-Armitage: **z = −3.73, p = 0.0002**):

| n_hop | % recovered |
|---|---|
| **2-hop** | **28.5%** |
| 3-hop | 18.1% |
| 4-hop | 19.4% |

Le 2-hop sono nettamente più recuperabili (+10 pp rispetto a 3/4-hop). Coerente
con l'intuizione: le domande 2-hop richiedono meno fatti incrociati, e il
rewriting riesce a riorganizzarli più facilmente. Il test di Cochran-Armitage
conferma che la probabilità di recupero decresce in modo monotonico con
la complessità.

**Content vs style:** nessuna differenza significativa (p = 0.31). La
distinzione tradizionale del design sperimentale non spiega questo fenomeno.

**Interazione hop × instruction:** non significativa (tutti i β di interazione
p > 0.1). Gli effetti sono additivi: una chain 2-hop con `elaborate` accumula
i vantaggi di entrambi (~31%), senza sinergia extra.

### Quando avviene il recupero e quanto dura?

Il **70.8% dei recuperi avviene già a step 1**, in modo uniforme tra istruzioni
e hop. Il recupero tardivo (step 2 o 3) è raro e non predetto da nessuna
variabile.

Sulla persistenza, tra le 394 chain recovered:
- **35.3%** mantiene F1 > 0 su tutti e 3 gli step successivi (recupero robusto)
- **23.1%** su 2 step su 3
- **41.6%** su 1 step su 3 soltanto (recupero transitorio)

La distribuzione di persistenza è simile tra istruzioni (chi² = 6.97, p = 0.32),
quindi non c'è un'istruzione che "guarisce in modo più stabile" delle altre.

> Il rewriting iterativo non è solo un processo distruttivo. In circa il 22%
> dei casi in cui il testo originale non bastava per rispondere, il rewriting
> produce una versione che invece funziona — soprattutto per domande semplici
> (2-hop) e con l'istruzione `elaborate`.

---

## 5. Limitazioni

| Limitazione | Impatto |
|---|---|
| OFS solo su 55 qid 2-hop (300q) | RQ1 (degrado fattualità) e RQ2b non concludibili sull'intero 300q |
| Answer F1 in 4-bit NF4, non bf16 | I valori assoluti potrebbero essere leggermente sottostimati; i confronti relativi restano validi |
| Self-Refine non eseguito | RQ3 resta aperta |
| Solo 3 step di rewriting | Non sappiamo se il trend si stabilizza o prosegue a step 4/5 |
| Pilot: n=15 | Solo effetti grandi rilevabili; il 300q è il dataset principale per le conclusioni |

---

## 6. Conclusioni in tre righe

> Il rewriting iterativo produce due fenomeni distinti: (1) un crollo di Answer F1
> quasi totalmente concentrato al primo step (−17 pp sul 300q), guidato in larga
> parte da compressione progressiva del testo e perdita di informazione; (2) un
> degrado di fattualità progressivo ma contenuto (OFS −3 pp in 3 step, significativo
> già sul pilot 15q). La lunghezza prodotta è il confondente principale: tutte le
> istruzioni comprimono aggressivamente (le medie vanno da 458 a 935 token contro
> i ~2340 dell'originale), e gran parte del degrado attribuito a step o istruzione
> è in realtà un effetto di questa compressione. Il processo non è però solo
> distruttivo: in ~22% dei casi inizialmente sbagliati, il rewriting riesce a
> "riparare" la risposta.

---

## 7. Mappa rapida dei file

```
results/
├── SUMMARY.md                          ← questo file
├── 15q/stats/
│   ├── README.md                       ← risultati completi pilot 15q
│   ├── diagnostics/
│   └── inference/
├── 300q/stats/
│   ├── README.md                       ← risultati completi 300q (senza OFS, vedi §2.8)
│   ├── diagnostics/
│   ├── inference/
│   ├── elaborate/
│   │   └── README.md                   ← analisi dedicata elaborate
│   └── recovery/
│       └── README.md                   ← analisi recupero F1
├── plots/
│   ├── 15q/
│   └── 300q/                           ← grafici di riferimento per le 300q
├── newsqa/                             ← pipeline avviata (run completi in corso)
└── fictionalqa/                        ← pipeline avviata (run completi in corso)
```
