# `elaborate` — cosa fa davvero (300q)

Analisi mirata dell'istruzione **elaborate** sul run 300q. Lo script che produce
questi file è
[`scripts/_common/elaborate_analysis.py`](../../../../scripts/_common/elaborate_analysis.py).

---

## TL;DR

> Su 300q, `elaborate` **non elabora**: produce sistematicamente testi più
> corti dell'originale (mediana 33% della lunghezza iniziale al primo step).
> Il **73% delle catene si dimezza** al primo step e il **32% scende sotto i
> 200 token** (collasso quasi totale). La lunghezza prodotta è la variabile
> più importante: catene "corte" (<200 token) hanno F1 quasi **dimezzato**
> (0.12 vs 0.22) e BERTScore −5 pt rispetto alle catene "lunghe". Quando si
> controlla per `n_tokens` nel modello, l'effetto residuo dell'istruzione su
> OFS quasi si annulla — **gran parte di quello che attribuivamo a "elaborate
> degrada la fattualità" è in realtà un effetto della compressione**.

---

## 1. Cosa fa elaborate ai token? (Blocco 1)

### 1.1 Lunghezza per step

**File:** [`01_elaborate_ntokens_by_step.csv`](01_elaborate_ntokens_by_step.csv)

| Step | n catene | media | mediana | min | max |
|---|---|---|---|---|---|
| 0 (originale) | 891 | **2445** | 2340 | 1302 | 4228 |
| 1 | 891 | **710** | 813 | 40 | 2048 |
| 2 | 891 | 647 | 703 | 38 | 1948 |
| 3 | 891 | 620 | 645 | 60 | 1829 |

→ Al primo step l'originale è già ridotto **al 29% della lunghezza**. È
l'opposto di elaborare.

### 1.2 Compliance — quanto spesso elaborate elabora davvero?

**File:** [`02_elaborate_compliance.csv`](02_elaborate_compliance.csv)

| Step | % catene che AUMENTANO la lunghezza | % che si DIMEZZANO | % CATASTROFICHE (<200 tok) |
|---|---|---|---|
| 1 | **0.0%** | 73.3% | 32.1% |
| 2 | **0.0%** | 83.6% | 27.6% |
| 3 | **0.0%** | 87.0% | 21.4% |

→ **Zero catene su 891** producono un testo più lungo dell'originale. Una su
tre crolla a meno di 200 token al primo step.

### 1.3 elaborate vs shorten — sono davvero diverse?

**File:** [`04_elaborate_vs_shorten_ntokens.csv`](04_elaborate_vs_shorten_ntokens.csv)

| Step | mean elaborate | mean shorten | mean_diff | p_holm |
|---|---|---|---|---|
| 1 | 710 | 458 | +252 | <0.001 |
| 2 | 647 | 352 | +296 | <0.001 |
| 3 | 619 | 308 | +312 | <0.001 |

→ Elaborate produce significativamente più token di shorten (differenza
robusta, p < 0.001 a ogni step), quindi **non sono identiche**. Ma elaborate
non aumenta mai la lunghezza rispetto all'originale: si comporta come "shorten
leggera".

---

## 2. elaborate vs altre istruzioni sulle metriche (Blocco 2)

**File:** [`10_elaborate_vs_other_metrics.csv`](10_elaborate_vs_other_metrics.csv)

Contrasti Wilcoxon paired (entro qid), aggregati su step ≥ 1.

| Metrica | Contrasto | mean elaborate | mean other | mean_diff | p_holm |
|---|---|---|---|---|---|
| **F1** | vs shorten | 0.189 | 0.185 | +0.003 | 0.37 |
| **F1** | vs formality | 0.189 | 0.206 | −0.017 | 1.00 |
| **F1** | vs paraphrase | 0.189 | 0.196 | −0.008 | 1.00 |
| **OFS** | vs shorten | 0.870 | 0.855 | +0.016 | 1.00 |
| **OFS** | vs formality | 0.870 | 0.884 | **−0.014** | **0.062 (marginale)** |
| **OFS** | vs paraphrase | 0.870 | 0.855 | +0.015 | 1.00 |
| **BERT base** | vs shorten | 0.839 | 0.836 | +0.003 | 0.08 |
| **BERT base** | vs formality | 0.839 | 0.872 | **−0.033** | **<0.001** |
| **BERT base** | vs paraphrase | 0.839 | 0.839 | +0.000 | 1.00 |

**Lettura:**
- elaborate **non è la peggiore** su F1 (è in mezzo, vicina alle altre).
- Su OFS, elaborate è leggermente più bassa di formality (−0.014, p marginale),
  in linea con le altre.
- Su BERTScore baseline, elaborate è chiaramente più lontana dall'originale
  rispetto a formality (−0.033, p < 0.001). Coerente: comprimendo a 1/3 della
  lunghezza, si perde molta sovrapposizione lessicale.

---

## 3. n_tokens come predittore: la lunghezza guida la qualità? (Blocco 3) ★

Questa è la parte interessante. Modelli misti **dentro elaborate**, controllando
per qid:

### 3.1 F1 dipende da quanto è lungo il testo?

**File:** [`20_f1_on_logtokens_elaborate.csv`](20_f1_on_logtokens_elaborate.csv)

`F1 ~ log(n_tokens) + step + (1|qid)`:

| Termine | Coef | p |
|---|---|---|
| log(n_tokens) | **+0.071** | <0.001 |
| step 2 | −0.024 | 0.046 |
| step 3 | −0.042 | <0.001 |

→ **Sì**: per ogni *raddoppio* della lunghezza prodotta, F1 cresce di ~0.05
punti. La lunghezza è un predittore robusto della qualità della risposta.

### 3.2 OFS dipende da quanto è lungo il testo?

**File:** [`21_ofs_on_logtokens_elaborate.csv`](21_ofs_on_logtokens_elaborate.csv)

`OFS ~ log(n_tokens) + step + (1|qid)`:

| Termine | Coef | p |
|---|---|---|
| log(n_tokens) | +0.007 | 0.24 (n.s.) |
| step 2 | −0.019 | 0.049 |
| step 3 | −0.036 | <0.001 |

→ Dentro elaborate, **la lunghezza non predice OFS** in modo significativo.
Il calo di OFS sembra venire da step, non dalla compressione *all'interno
delle catene elaborate*.

### 3.3 Mediation: l'effetto di instruction su OFS si attenua se controllo per n_tokens?

**File:** [`22_ofs_on_instruction_no_tokens.csv`](22_ofs_on_instruction_no_tokens.csv) e [`23_ofs_on_instruction_with_tokens.csv`](23_ofs_on_instruction_with_tokens.csv)

| Termine | Senza n_tokens | Con n_tokens |
|---|---|---|
| formality vs elaborate | +0.014 (p = 0.04) | −0.003 (p = 0.66) |
| paraphrase vs elaborate | −0.015 (p = 0.02) | +0.006 (p = 0.35) |
| shorten vs elaborate | −0.016 (p = 0.01) | +0.012 (p = 0.06) |
| log(n_tokens) | — | **+0.060 (p < 0.001)** |

**Il coefficiente di instruction_type passa da significativo a ~zero** quando
si aggiunge `log(n_tokens)` al modello. **Interpretazione: la differenza tra
istruzioni su OFS è quasi interamente spiegata dalla lunghezza prodotta, non
dall'istruzione in sé.**

Detto altrimenti: due testi della stessa lunghezza hanno OFS simile,
*indipendentemente da quale istruzione li ha generati*. È la lunghezza che
conta.

---

## 4. Failure mode: catene "corte" vs "lunghe" (Blocco 4)

**File:** [`30_failure_short_vs_long.csv`](30_failure_short_vs_long.csv)

Separiamo le catene elaborate in "corte" (<200 token, catastrofiche) e "lunghe"
(≥200 token), e confrontiamo le metriche con Mann-Whitney.

| Metrica | n corte | n lunghe | mean corte | mean lunghe | mean_diff | p_holm |
|---|---|---|---|---|---|---|
| **F1** | 723 | 1950 | **0.117** | **0.216** | **−0.099** | <0.001 |
| **OFS** | 107 | 379 | 0.865 | 0.872 | −0.007 | 0.033 |
| **BERT base** | 723 | 1950 | **0.801** | **0.853** | **−0.052** | <0.001 |

→ Le catene corte hanno **F1 quasi dimezzato** (0.117 vs 0.216) e BERTScore
**5 punti sotto** le lunghe. **L'OFS invece resta sorprendentemente stabile**
(differenza −0.007, statisticamente rilevabile per via degli n grandi ma
trascurabile praticamente).

Conclusione: **se elaborate collassa in pochi token, la fattualità del poco
che resta è ancora alta, ma diventa inutile per il task QA** (mancano le
informazioni per rispondere).

### 4.1 Dove si concentrano le catene "corte"?

**File:** [`31_short_rate_by_hop_step.csv`](31_short_rate_by_hop_step.csv)

| n_hop | step 1 | step 2 | step 3 |
|---|---|---|---|
| 2 | 29% | 25% | 18% |
| 3 | 27% | 24% | 20% |
| **4** | **40%** | **33%** | **26%** |

→ Le domande **4-hop** sono particolarmente vulnerabili al collasso (40% di
catene corte al primo step, vs ~28% di 2/3-hop). I testi originali delle
4-hop sono i più lunghi e complessi, e il modello sembra trattarli più
aggressivamente.

---

## 5. Take-away per il paper

1. **L'etichetta "elaborate" è ingannevole**: di fatto si comporta come una
   forma di compressione blanda. Su 891 catene, *zero* superano la lunghezza
   originale al primo step.
2. **Il driver del degrado è la lunghezza, non l'istruzione**: il modello
   mediation mostra che gli effetti di instruction su OFS svaniscono quando
   si controlla per n_tokens.
3. **Le 4-hop sono le più colpite dal collasso** di elaborate (40% catene
   corte vs 28% di 2/3-hop al primo step).
4. **F1 segue la lunghezza**: catene corte → F1 dimezzato. È coerente con
   "meno informazione disponibile → meno chance di rispondere".

---

## 6. Mappa dei file

```
results/300q/stats/elaborate/
├── README.md                                  ← questo file
├── 01_elaborate_ntokens_by_step.csv           ← distribuzione lunghezze
├── 02_elaborate_compliance.csv                ← % aumento / dimezzamento / collasso
├── 03_ntokens_by_instruction_step.csv         ← confronto con altre istruzioni
├── 04_elaborate_vs_shorten_ntokens.csv        ← Wilcoxon
├── 10_elaborate_vs_other_metrics.csv          ← Blocco 2
├── 20_f1_on_logtokens_elaborate.csv           ← Blocco 3
├── 21_ofs_on_logtokens_elaborate.csv
├── 22_ofs_on_instruction_no_tokens.csv        ← mediation: prima
├── 23_ofs_on_instruction_with_tokens.csv      ← mediation: dopo
├── 30_failure_short_vs_long.csv               ← Blocco 4
├── 31_short_rate_by_hop_step.csv
└── 32_short_rate_by_group_step.csv
```
