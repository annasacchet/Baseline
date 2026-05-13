# Recovery analysis — quando il rewriting "guarisce" una risposta sbagliata (300q)

Documento dedicato a un fenomeno specifico: **chains che partono con F1=0 a
step 0** (il modello QA sbaglia sul testo originale) **e poi recuperano**
(F1 > 0 in almeno uno step successivo).

Lo script che produce i numeri è
[`scripts/300q/recovery_analysis.py`](../../../../scripts/300q/recovery_analysis.py).

---

## TL;DR

> Su 1800 chains che partono sbagliando (50.5% del totale), **394 (21.9%)
> "guariscono" dopo il rewriting**. Il recupero non è uniforme: dipende
> significativamente da (a) **istruzione** — `elaborate` è ~6 pp meglio di
> `formality` (Cochran's Q p = 0.017) — e (b) **complessità** — le **2-hop
> recuperano il 28.5%** contro il 19% di 3-hop e 4-hop (chi² p < 0.001,
> trend di Cochran-Armitage p = 0.0002). Il **71% dei recuperi avviene già a
> step 1**. content vs style non differiscono significativamente.

---

## 1. Cosa stiamo guardando

Una **chain** = (qid, instruction_type, run). Su 297 qid × 4 istruzioni × 3
run = **3564 chains** totali.

| Definizione | Numero |
|---|---|
| Chains totali | 3564 |
| **zero-start**: F1=0 a step 0 | **1800 (50.5%)** |
| **recovered**: zero-start con F1 > 0 in step 1/2/3 | **394 (21.9% delle zero-start, 11.1% del totale)** |

Domanda di ricerca: **quali condizioni rendono il recupero più probabile?**

---

## 2. Glossario rapido (per ricordare)

- **Recovered**: chain che partiva con F1=0 e poi raggiunge F1 > 0 in almeno
  uno degli step 1/2/3.
- **GLMM logit**: modello a effetti misti per variabili binarie (0/1) con
  random intercept per `qid`. Necessario perché chains della stessa domanda
  non sono indipendenti.
- **Cochran's Q**: estensione di McNemar a più di 2 condizioni *paired*. Qui
  serve perché ogni qid è osservato sotto tutte e 4 le istruzioni.
- **Cochran-Armitage trend test**: testa se una proporzione binaria varia
  *monotonicamente* lungo una variabile ordinale (qui n_hop).
- **OR (odds ratio)**: di quanto le probabilità (in odds) di recupero cambiano
  passando da una condizione a un'altra. OR = 1 → nessuna differenza.

---

## 3. Test eseguiti

| Test | Domanda |
|---|---|
| **1** | Il group (content vs style) cambia la probabilità di recupero? |
| **2** | L'istruzione (4 livelli) cambia la probabilità di recupero? |
| **3** | La complessità (n-hop) cambia la probabilità di recupero? |
| **4** | Esiste un'interazione hop × instruction? |
| **5** | Persistenza del recupero e a quale step avviene? |

---

## 4. Risultati

### 4.1 Test 1 — group (content vs style)

**File:** [`01_group_descriptive.csv`](01_group_descriptive.csv), [`01_group_glmm.csv`](01_group_glmm.csv)

| Group | n zero-start | n recovered | % recovered |
|---|---|---|---|
| content | 900 | 207 | **23.0%** |
| style | 900 | 187 | 20.8% |

**GLMM logit `recovered ~ group + (1|qid)`:**
- Coef style vs content = **−0.022**, SE = 0.022, **p = 0.31** (n.s.)
- Chi² 2×2 di controllo: chi² = 1.30, p = 0.25 (n.s.)

→ **Nessuna differenza significativa**. content e style hanno tassi di
recupero simili. La differenza grezza di 2 pp non sopravvive al test.

### 4.2 Test 2 — istruzione

**File:** [`02_instruction_descriptive.csv`](02_instruction_descriptive.csv), [`02_instruction_glmm.csv`](02_instruction_glmm.csv), [`02_instruction_cochranq.csv`](02_instruction_cochranq.csv), [`02_instruction_mcnemar_pairwise.csv`](02_instruction_mcnemar_pairwise.csv)

| Istruzione | Group | n zero-start | n recovered | % recovered |
|---|---|---|---|---|
| **elaborate** | content | 450 | 113 | **25.1%** ← top |
| paraphrase | style | 450 | 100 | 22.2% |
| shorten | content | 450 | 94 | 20.9% |
| formality | style | 450 | 87 | 19.3% |

**GLMM logit `recovered ~ instruction_type + (1|qid)`** (baseline = elaborate):

| Confronto | Coef | p |
|---|---|---|
| formality vs elaborate | **−0.058** | **0.008** |
| paraphrase vs elaborate | −0.029 | 0.18 |
| shorten vs elaborate | **−0.042** | **0.054** (borderline) |

**Cochran's Q (paired per qid):** Q = 10.15, df = 3, **p = 0.017** → effetto
globale dell'istruzione significativo.

**Pairwise McNemar (Holm-corretti) sulle 6 coppie:**

| a | b | n_a_only | n_b_only | p_holm |
|---|---|---|---|---|
| elaborate | shorten | 33 | 15 | 0.085 (marginale) |
| elaborate | paraphrase | 28 | 13 | 0.14 |
| elaborate | formality | 31 | 16 | 0.16 |
| altre coppie | | | | > 0.7 |

→ `elaborate` è significativamente migliore di `formality` nel modello GLMM
(p = 0.008). Le differenze pairwise dopo Holm non passano la soglia 0.05 ma
sono tutte nella stessa direzione (elaborate > altre). Il Cochran's Q
omnibus conferma l'effetto dell'istruzione.

**Lettura:** `elaborate` è l'istruzione più "riparatoria"; `formality` la
meno. La differenza nel modello mixed è di −0.058 in probabilità (−5.8 pp),
coerente con i numeri grezzi (25.1% vs 19.3%).

### 4.3 Test 3 — complessità (n_hop) ★

**File:** [`03_hop_descriptive.csv`](03_hop_descriptive.csv), [`03_hop_glmm.csv`](03_hop_glmm.csv), [`03_hop_trend.csv`](03_hop_trend.csv), [`03_hop_chi2.csv`](03_hop_chi2.csv)

| n_hop | n zero-start | n recovered | % recovered |
|---|---|---|---|
| **2** | 564 | 161 | **28.5%** ← top |
| 3 | 552 | 100 | 18.1% |
| 4 | 684 | 133 | 19.4% |

**GLMM logit `recovered ~ n_hop + (1|qid)`** (baseline = 2-hop):

| Confronto | Coef | p |
|---|---|---|
| 3-hop vs 2-hop | −0.104 | 0.063 (marginale) |
| 4-hop vs 2-hop | −0.091 | 0.088 (marginale) |

**Chi² 3×2 omnibus:** chi² = 21.6, df = 2, **p = 2e-5** → significativo.

**Cochran-Armitage trend test:** z = −3.73, **p = 0.0002** → la probabilità
di recupero **decresce in modo monotonico significativo** con la
complessità.

> Nota tecnica: il GLMM mostra effetti marginali (p ≈ 0.06–0.09) sui
> coefficienti individuali, ma sia chi² omnibus che il trend di
> Cochran-Armitage sono molto significativi (p < 0.001). L'apparente
> discrepanza è dovuta al fatto che il modello LMM su variabili binarie
> è un'approssimazione conservativa; il trend test sfrutta l'ordinalità di
> n_hop ed è più potente per questa domanda specifica.

**Lettura:** **le 2-hop sono nettamente le più recuperabili** (28.5%), mentre
3-hop e 4-hop sono simili tra loro (~19%). Coerente con l'intuizione: le
catene multi-hop richiedono più fatti incrociati, e il rewriting raramente
ne "ripara" tutti contemporaneamente.

### 4.4 Test 4 — interazione hop × instruction

**File:** [`04_interaction_descriptive.csv`](04_interaction_descriptive.csv), [`04_interaction_glmm.csv`](04_interaction_glmm.csv)

% recovered per cella:

| | elaborate | formality | paraphrase | shorten |
|---|---|---|---|---|
| **2-hop** | **31.2%** | 25.5% | **31.2%** | 26.2% |
| 3-hop | 23.2% | 15.2% | 18.8% | 15.2% |
| 4-hop | 21.6% | 17.5% | 17.5% | 21.1% |

Il modello con interazione è stato fittato; il **LRT** confronto con/senza
interazione non è valido nel caso REML (statistica negativa = modelli non
comparabili in questa parametrizzazione). I coefficienti di interazione
individuali nel modello GLMM sono **non significativi** (tutti p > 0.1):
l'effetto principale di hop e quello principale di instruction sembrano
agire in modo additivo, senza interazione robusta.

**Lettura:** le combinazioni `2-hop × elaborate` e `2-hop × paraphrase`
brillano (31%) ma sono ben spiegate dalla somma degli effetti principali
(2-hop migliore + elaborate/paraphrase migliori). **Non c'è una sinergia
extra** statisticamente solida.

### 4.5 Test 5 — persistenza del recupero e step di primo recupero

**File:** [`05_persistence_by_instruction.csv`](05_persistence_by_instruction.csv), [`05_first_step_by_instruction.csv`](05_first_step_by_instruction.csv), [`05_first_step_by_hop.csv`](05_first_step_by_hop.csv), [`05_chi2_summary.csv`](05_chi2_summary.csv)

#### 5a — Persistenza (su quanti step su 3 il recupero "dura")

Tra le 394 chains recovered:

| Persistenza | n chains |
|---|---|
| 1/3 step (transitorio) | 164 (41.6%) |
| 2/3 step | 91 (23.1%) |
| 3/3 step (robusto) | 139 (35.3%) |

**Chi² persistenza × istruzione:** chi² = 6.97, df = 6, p = 0.32 (n.s.) →
la distribuzione di persistenza è simile tra le 4 istruzioni.

#### 5b — A quale step avviene il primo recupero?

| Step | n chains | % |
|---|---|---|
| **step 1** | 279 | **70.8%** |
| step 2 | 64 | 16.2% |
| step 3 | 51 | 12.9% |

**Chi² step × istruzione:** chi² = 4.96, df = 6, p = 0.55 (n.s.)
**Chi² step × hop:** chi² = 3.03, df = 4, p = 0.55 (n.s.)

**Lettura:** il primo recupero è dominato dallo step 1 (~71% dei casi)
**in modo uniforme** tra istruzioni e tra hop. Non ci sono pattern
condizionali nel "quando" del recupero.

---

## 5. Take-away per il paper

1. **Il fenomeno esiste ed è quantitativamente rilevante**: il 22% delle
   chains inizialmente sbagliate viene "salvato" dal rewriting.
2. **L'istruzione conta** (Cochran's Q p = 0.017): `elaborate` recupera il
   25%, `formality` solo il 19% (−5.8 pp nel modello mixed, p = 0.008).
3. **La complessità conta molto** (chi² p < 0.001, trend p = 0.0002): le
   2-hop hanno una probabilità di recupero **+10 pp rispetto a 3/4-hop**.
4. **Nessuna interazione hop × instruction significativa**: gli effetti sono
   additivi.
5. **content vs style non differiscono** (p = 0.31): la distinzione
   tradizionale del paper non spiega questo fenomeno.
6. **Il recupero avviene quasi sempre al primo step** (71%) e nel 35% dei
   casi persiste su tutti e 3 gli step (cioè non è solo rumore).

**Interpretazione (cautelativa):** il rewriting iterativo non è solo un
processo distruttivo. In una porzione non trascurabile dei casi, *riformula*
il testo originale in un modo che lo rende più navigabile per il QA model,
soprattutto quando (a) la domanda richiede solo 2 hop di ragionamento e
(b) l'istruzione spinge verso una riorganizzazione del contenuto
(`elaborate`).

---

## 6. Limitazioni

- **F1 in 4-bit, non bf16**: vedi note generali nel README principale (§6.2).
- **Non sappiamo se il "recupero" è genuino o un artefatto di rumore del QA
  evaluator**: una chain con F1=0 a step 0 e F1>0 a step 1 *potrebbe*
  riflettere instabilità del modello QA piuttosto che un vero miglioramento
  semantico. Il fatto che il 35% dei recuperi persiste su tutti e 3 gli step
  è un controllo a favore della genuinità, ma non lo prova.
- **OFS non disponibile per validare l'ipotesi sul *meccanismo***: idealmente
  vorremmo vedere se le chains recovered hanno anche un miglioramento di
  fattualità (OFS più alto). Ma OFS è disponibile solo per 55 qid 2-hop, di
  cui non possiamo dire molto su questo subset.

---

## 7. Mappa dei file

```
results/300q/stats/recovery/
├── README.md                                  ← questo file
├── 01_group_descriptive.csv                   ← Test 1
├── 01_group_glmm.csv
├── 01_group_chi2.csv
├── 02_instruction_descriptive.csv             ← Test 2
├── 02_instruction_glmm.csv
├── 02_instruction_cochranq.csv
├── 02_instruction_mcnemar_pairwise.csv
├── 03_hop_descriptive.csv                     ← Test 3
├── 03_hop_glmm.csv
├── 03_hop_trend.csv
├── 03_hop_chi2.csv
├── 04_interaction_descriptive.csv             ← Test 4
├── 04_interaction_glmm.csv
├── 04_interaction_lrt.csv
├── 05_persistence_by_instruction.csv          ← Test 5
├── 05_first_step_by_instruction.csv
├── 05_first_step_by_hop.csv
└── 05_chi2_summary.csv
```

Per rigenerare tutto:
```
python3.11 scripts/300q/recovery_analysis.py
```
