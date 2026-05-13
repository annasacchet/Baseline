# `elaborate` — cosa fa davvero (15q, pilot)

Analisi mirata dell'istruzione **elaborate** sul pilot 15q. Lo script che
produce questi file è
[`scripts/_common/elaborate_analysis.py`](../../../../scripts/_common/elaborate_analysis.py).

---

## TL;DR

> Su 15q, `elaborate` **sostanzialmente non elabora**: solo l'11–16% delle
> catene aumenta la lunghezza rispetto all'originale (mediana 68–72% della
> lunghezza iniziale). **Nessuna catena collassa sotto i 200 token** (a
> differenza di quanto succede sul 300q), ma il pilot è troppo piccolo
> (n = 45 catene elaborate per step) per analizzare failure mode. Il dato
> interessante: dentro elaborate, **più tokens → meno OFS** (β = −0.137,
> p < 0.001) — comportamento opposto al 300q. E la mediation mostra che il
> "vantaggio" delle altre istruzioni su OFS rispetto a elaborate è
> parzialmente spiegato dalla loro maggior brevità (testi più corti, più
> facili da verificare per i fatti).

---

## 1. Cosa fa elaborate ai token? (Blocco 1)

### 1.1 Lunghezza per step

**File:** [`01_elaborate_ntokens_by_step.csv`](01_elaborate_ntokens_by_step.csv)

| Step | n catene | media | mediana | min | max |
|---|---|---|---|---|---|
| 0 (originale) | 45 | **2311** | 2357 | 1692 | 3276 |
| 1 | 45 | 1506 | 1488 | 789 | 2048 |
| 2 | 45 | 1543 | 1535 | 679 | 2048 |
| 3 | 45 | **1638** | 1685 | 701 | 2048 |

→ Al primo step la lunghezza scende al 65% dell'originale, ma **risale
gradualmente verso step 3** (71%). Non è la compressione drammatica del
300q, ma neanche un'elaborazione reale. Da notare il tetto **2048 token**:
diverse catene saturano il limite di output del modello.

### 1.2 Compliance — quanto spesso elaborate elabora davvero?

**File:** [`02_elaborate_compliance.csv`](02_elaborate_compliance.csv)

| Step | % AUMENTANO la lunghezza | % si DIMEZZANO | % CATASTROFICHE (<200 tok) |
|---|---|---|---|
| 1 | **11.1%** | 20.0% | 0% |
| 2 | 11.1% | 15.6% | 0% |
| 3 | **15.6%** | 13.3% | 0% |

→ Una minoranza di catene aumenta la lunghezza, e il pattern **migliora con
gli step** (15.6% a step 3). Nessun collasso catastrofico.

### 1.3 elaborate vs shorten — sono davvero diverse?

**File:** [`04_elaborate_vs_shorten_ntokens.csv`](04_elaborate_vs_shorten_ntokens.csv)

| Step | mean elaborate | mean shorten | mean_diff | p_holm |
|---|---|---|---|---|
| 1 | 1506 | 603 | +903 | 0.0002 |
| 2 | 1543 | 505 | +1038 | 0.0002 |
| 3 | 1638 | 464 | +1175 | 0.0002 |

→ elaborate produce circa **3× i token di shorten** — quindi le due istruzioni
sono ben separate in lunghezza, anche se elaborate non aumenta mai sopra
l'originale.

---

## 2. elaborate vs altre istruzioni sulle metriche (Blocco 2)

**File:** [`10_elaborate_vs_other_metrics.csv`](10_elaborate_vs_other_metrics.csv)

Contrasti Wilcoxon paired (entro qid), aggregati su step ≥ 1.

| Metrica | Contrasto | mean elaborate | mean other | mean_diff | p_holm |
|---|---|---|---|---|---|
| **F1** | vs shorten | 0.347 | 0.315 | +0.032 | 1.00 |
| **F1** | vs formality | 0.347 | 0.338 | +0.009 | 1.00 |
| **F1** | vs paraphrase | 0.347 | 0.350 | −0.003 | 1.00 |
| **OFS** | vs shorten | 0.842 | 0.896 | **−0.055** | 0.075 (marginale) |
| **OFS** | vs formality | 0.842 | 0.900 | **−0.059** | **0.016** |
| **OFS** | vs paraphrase | 0.842 | 0.908 | **−0.067** | **0.002** |
| **BERT base** | vs shorten | 0.888 | 0.835 | **+0.053** | <0.001 |
| **BERT base** | vs formality | 0.888 | 0.902 | −0.013 | 0.10 |
| **BERT base** | vs paraphrase | 0.888 | 0.868 | **+0.020** | **0.008** |

**Lettura:**
- Su **F1** elaborate è equivalente alle altre — nessuna differenza
  significativa.
- Su **OFS**, elaborate è chiaramente la **peggiore**: −5.5/−6.7 punti contro
  le altre tre. Significativo dopo correzione.
- Su **BERTScore baseline**, elaborate è **più simile** all'originale rispetto
  a shorten e paraphrase (mantiene più sovrapposizione lessicale, coerente
  con il fatto che è più lunga).

---

## 3. n_tokens come predittore: la lunghezza guida la qualità? (Blocco 3) ★

### 3.1 F1 dipende da quanto è lungo il testo?

**File:** [`20_f1_on_logtokens_elaborate.csv`](20_f1_on_logtokens_elaborate.csv)

`F1 ~ log(n_tokens) + step + (1|qid)`:

| Termine | Coef | p |
|---|---|---|
| log(n_tokens) | +0.066 | 0.52 (n.s.) |
| step 2 | +0.011 | 0.78 |
| step 3 | −0.012 | 0.77 |

→ Con n = 45 dentro elaborate, **non rileviamo** un effetto di lunghezza su
F1. Direzione positiva (più token → F1 più alto), magnitudo simile al 300q,
ma p > 0.05 per mancanza di potenza.

### 3.2 OFS dipende da quanto è lungo il testo?

**File:** [`21_ofs_on_logtokens_elaborate.csv`](21_ofs_on_logtokens_elaborate.csv)

`OFS ~ log(n_tokens) + step + (1|qid)`:

| Termine | Coef | p |
|---|---|---|
| log(n_tokens) | **−0.137** | **<0.001** |
| step 2 | −0.023 | 0.013 |
| step 3 | −0.054 | <0.001 |

→ **All'interno di elaborate, testi più lunghi hanno OFS più bassa.**
Coefficiente negativo grande (β = −0.137 per ogni log unit di token).

**Interpretazione plausibile:** OpenFActScore valuta i fatti decomposti dal
testo riscritto. Più il testo è lungo, più fatti vengono estratti, e in
media una frazione maggiore di questi può essere non supportata o
contraddetta. È un effetto noto della metrica: testi più lunghi tendono a
ottenere OFS più bassi non perché siano "meno corretti", ma perché la
metrica è più stringente quando ci sono più asserzioni da verificare.

> Va dichiarato come limite metrico, non come fatto sostanziale sul rewriting.

### 3.3 Mediation: l'effetto di instruction su OFS si attenua se controllo per n_tokens?

**File:** [`22_ofs_on_instruction_no_tokens.csv`](22_ofs_on_instruction_no_tokens.csv) e [`23_ofs_on_instruction_with_tokens.csv`](23_ofs_on_instruction_with_tokens.csv)

Coefficienti rispetto a elaborate (baseline):

| Termine | Senza n_tokens | Con n_tokens |
|---|---|---|
| formality vs elaborate | **+0.059 (p < 0.001)** | +0.050 (p < 0.001) |
| paraphrase vs elaborate | **+0.067 (p < 0.001)** | +0.042 (p < 0.001) |
| shorten vs elaborate | **+0.055 (p < 0.001)** | +0.009 (p = 0.41) |
| log(n_tokens) | — | **−0.041 (p < 0.001)** |

**Cosa si vede:**
- Senza controllare per la lunghezza, **tutte e tre** le altre istruzioni hanno
  OFS più alta di elaborate.
- Aggiungendo `log(n_tokens)`: la differenza tra elaborate e shorten si
  **annulla** (da +0.055 a +0.009, n.s.) — è interamente spiegata dalla
  brevità di shorten.
- La differenza con formality e paraphrase si attenua ma resta significativa
  (+0.050, +0.042): qui c'è un effetto dell'istruzione *oltre* la lunghezza.

**Take-away:** parte del "vantaggio" delle altre istruzioni su OFS è un
artefatto della loro maggior brevità (tendenza della metrica), non un vero
miglioramento di fattualità.

---

## 4. Failure mode (Blocco 4)

**File:** [`30_failure_short_vs_long.csv`](30_failure_short_vs_long.csv)

Sul 15q **nessuna catena elaborate scende sotto i 200 token**: la soglia di
"catastrofica" non scatta. Quindi l'analisi failure mode è vuota per questo
dataset.

| n_hop | % catene <200 tok |
|---|---|
| 2 | 0% |
| 3 | 0% |
| 4 | 0% |

> Sul 300q la stessa analisi mostra il 21–32% di catene collassate. La
> differenza tra i due dataset è notevole e meriterebbe un'investigazione
> separata (modelli diversi? prompt template aggiornato? selezione dei qid?).

---

## 5. Limitazione importante: saturazione del cap `max_new_tokens=2048`

Sul 15q, una porzione non trascurabile delle catene `elaborate` raggiunge il
**tetto di output del modello** (`n_tokens = 2048`). Questo significa che il
rewriting è stato **troncato** e il testo finale è incompleto.

**Saturazione (n_tokens ≥ 2048) per step, solo elaborate:**

| Step | n catene saturate | % saturate |
|---|---|---|
| 1 | 1 / 45 | 2.2% |
| 2 | 6 / 45 | 13.3% |
| 3 | **11 / 45** | **24.4%** |

**Per hop:**

| n_hop | % catene saturate (totale step ≥ 1) |
|---|---|
| 2 | 2.2% |
| **3** | **28.9%** |
| 4 | 8.9% |

**Confronto con le altre istruzioni** (sempre su 15q, step ≥ 1):

| Istruzione | n saturate / totale |
|---|---|
| **elaborate** | **18 / 135 (13.3%)** |
| formality | 0 / 135 (0%) |
| paraphrase | 0 / 135 (0%) |
| shorten | 0 / 135 (0%) |

**Solo elaborate satura il cap.** Le altre istruzioni producono testi
sufficientemente corti da non incontrare mai il limite.

### Conseguenze

1. **I confronti elaborate vs altre istruzioni sulle metriche sono distorti**:
   ~13% delle catene elaborate ha un testo tagliato a metà (e fino al 24% a
   step 3). I fatti del finale del testo sono persi → OFS e F1 sottostimati
   per quelle catene.
2. **La traiettoria 1→2→3 di elaborate è alterata**: a step 3 una catena su 4
   è troncata, quindi i numeri "elaborate a step 3" sono più rumorosi di
   quanto sembrino.
3. **Il problema NON esiste sul 300q** (saturazione 0.4% complessivo, solo a
   step 1), perché elaborate sul 300q comprime aggressivamente (vedi README
   del 300q).

### Come va dichiarato nel paper

> Sul pilot 15q, il 13.3% delle catene `elaborate` (24.4% a step 3) raggiunge
> il tetto `max_new_tokens = 2048`. Le metriche di fattualità e Answer F1 di
> queste catene sono calcolate su testi troncati. I confronti elaborate vs
> altre istruzioni vanno letti con questa cautela. Il problema è specifico al
> 15q ed elaborate: sul run 300q la saturazione è <0.5%, e nessun'altra
> istruzione satura mai il cap.

Una soluzione pulita richiederebbe di rilanciare il rewriting con
`max_new_tokens ≥ 4096`; in alternativa, una sensitivity analysis che
escluda le catene saturate. Né è stata fatta in questa analisi.

---

## 6. Take-away per il paper

1. **Compliance bassa ma migliore del 300q**: solo l'11–16% delle catene
   elaborate aumenta davvero la lunghezza, ma nessuna collassa.
2. **OFS elaborate < OFS altre istruzioni** (−5/−7 pt, significativo).
   La mediation mostra che parte di questo gap è artefatto della lunghezza
   (testi lunghi → OFS più bassa per costruzione della metrica).
3. **Direzione opposta al 300q sull'effetto n_tokens su OFS** (β negativo
   sul 15q, β ≈ 0 sul 300q). Da indagare: probabilmente è una caratteristica
   della metrica OFS quando i testi sono lunghi (tanti fatti = più chance di
   trovare almeno uno non supportato).
4. **Pilot troppo piccolo per failure mode**: con n=45 per step, non vediamo
   collassi né effetti deboli (es. log_ntokens su F1).

---

## 7. Mappa dei file

```
results/15q/stats/elaborate/
├── README.md                                  ← questo file
├── 01_elaborate_ntokens_by_step.csv
├── 02_elaborate_compliance.csv
├── 03_ntokens_by_instruction_step.csv
├── 04_elaborate_vs_shorten_ntokens.csv
├── 10_elaborate_vs_other_metrics.csv
├── 20_f1_on_logtokens_elaborate.csv
├── 21_ofs_on_logtokens_elaborate.csv
├── 22_ofs_on_instruction_no_tokens.csv
├── 23_ofs_on_instruction_with_tokens.csv
├── 30_failure_short_vs_long.csv               ← vuoto sul 15q
├── 31_short_rate_by_hop_step.csv              ← tutto 0%
└── 32_short_rate_by_group_step.csv            ← tutto 0%
```
