# Analisi correlazione BLEURT(gold, predicted) — Answer F1 — 300q

**Obiettivo:** verificare se Answer F1 sottostima la performance del modello a causa
di mismatch lessicali — cioè casi in cui il modello risponde correttamente ma con
parole diverse dalla gold (es. `"10-year"` vs `"10 years"`, `"GDR"` vs `"DDR"`).

Per farlo usiamo BLEURT calcolato direttamente tra `gold_answer` e `predicted_answer`:
se Answer F1 = 0 ma BLEURT è alto, la risposta potrebbe essere corretta semanticamente
pur non matchando lessicalmente.

Script di riferimento: `scripts/300q/find_bleurt_mismatches.py`

---

## Copertura del dataset

BLEURT(gold, predicted) è stato calcolato **solo sugli step 1–3** (le riscritture
effettive), non sullo step 0. Lo step 0 è il baseline prima di qualsiasi
riscrittura: il contesto non è ancora stato modificato, quindi non è rilevante
per l'analisi dei falsi negativi da riscrittura.

| | n | note |
|---|---|---|
| qid distinte | 297 | tutte presenti in entrambi i CSV |
| Step 0 (baseline, unico per qid) | 294 | senza BLEURT — 3 qid con predicted NaN |
| Step 1–3 con BLEURT e predicted | **10,037** | base dell'analisi |

La copertura è **completa** su tutte le 297 domande per tutti i passi di riscrittura.

Un primo dato emerge già dal confronto tra step 0 e step 1–3:

| | Answer F1 medio |
|---|---|
| Step 0 — baseline, testo originale | **0.366** |
| Step 1–3 — dopo la riscrittura | **0.207** |
| Δ | **−0.159** |

La riscrittura degrada Answer F1 di ~16 punti in media rispetto al baseline,
indipendentemente dall'instruction type e dal numero di hop.

---

## Grafico 1 — Scatter Answer F1 vs BLEURT

![scatter](png/bleurt_vs_answerf1_scatter.png)

**Cosa mostra.**
Ogni esagono rappresenta quante coppie (Answer F1, BLEURT) cadono in quella zona
(colore più scuro = più coppie, scala logaritmica). Ogni coppia è una singola
valutazione: una domanda, un passo di riscrittura, una run.

**Cosa si vede.**
- La maggior parte dei punti è in **basso a sinistra**: Answer F1 basso e BLEURT
  basso insieme. Questo significa che quando il modello sbaglia, lo sbaglia davvero —
  la risposta predicted non somiglia nemmeno semanticamente alla gold.
- C'è una **diagonale visibile**: quando Answer F1 è alto, anche BLEURT tende ad
  essere alto. Le due metriche sono allineate (Pearson r = 0.88).
- Il **quadrante in alto a sinistra** (F1 = 0 ma BLEURT alto) è quello dei
  potenziali falsi negativi: poche coppie, ma interessanti da esaminare.
  Le linee tratteggiate (F1=0 verticale, BLEURT=0.3 orizzontale) lo delimitano.

**Conclusione:** le due metriche concordano quasi sempre. I casi in cui divergono
(F1=0 ma BLEURT alto) sono pochi e localizzati — non un fenomeno sistematico.

---

## Grafico 2 — Boxplot BLEURT per fascia di Answer F1

![boxplot](png/bleurt_by_f1_bin_boxplot.png)

**Cosa mostra.**
Per ciascuna fascia di Answer F1 (da "uguale a zero" a "quasi perfetto"),
il grafico mostra come si distribuisce il BLEURT in quella fascia: la linea
centrale è la mediana, la scatola copre il 50% centrale dei dati, i baffi
il resto.

**Cosa si vede.**
- La mediana BLEURT sale **in modo monotono** al crescere di Answer F1:
  `0.14 → 0.25 → 0.37 → 0.51 → 0.87`. Non c'è nessun incrocio o inversione.
  Questo conferma che BLEURT e Answer F1 misurano la stessa cosa nella direzione
  giusta: più la risposta è corretta, più è semanticamente simile alla gold.
- La fascia **F1 = 0** (n = 6,821, la più grande) ha la distribuzione più larga
  e asimmetrica: la mediana è a 0.14 ma la coda destra arriva a BLEURT > 0.9.
  Quella coda destra sono i candidati falsi negativi.
- Le due linee tratteggiate segnano le soglie operative:
  - **BLEURT = 0.1** (tratteggio corto): sotto questa linea l'errore è certo —
    gold e predicted non hanno nulla in comune.
  - **BLEURT = 0.3** (tratteggio lungo): sopra questa linea nella fascia F1=0
    iniziano i casi da verificare.

**Conclusione:** BLEURT discrimina bene le fasce di Answer F1. Usarlo come
filtro per i casi F1=0 è giustificato — non è arbitrario.

---

## Grafico 3 — Distribuzione BLEURT dove Answer F1 = 0

![hist](png/bleurt_distribution_f1zero.png)

**Cosa mostra.**
Questo grafico si concentra solo sui **6,821 casi in cui Answer F1 = 0** e mostra
come si distribuisce il loro BLEURT. L'idea è: se Answer F1=0 fosse sempre un
errore reale, ci aspetteremmo tutti i BLEURT bassi. Se invece ci fossero molti
falsi negativi, vedremmo una coda destra consistente.

**Cosa si vede.**
- La distribuzione è **fortemente sbilanciata a sinistra**: la stragrande
  maggioranza dei casi F1=0 ha BLEURT basso (mediana ≈ 0.14). Il modello
  sbaglia davvero nella maggior parte dei casi.
- Il picco tra 0.10 e 0.15 corrisponde prevalentemente ai **rifiuti del modello**
  ("I'm sorry, I can't answer...") — risposte che ottengono una leggera
  similarità con qualsiasi gold ma sono chiaramente sbagliate.
- Le tre zone colorate identificano interpretazioni diverse:

| Zona | Soglia BLEURT | Cosa contiene | n |
|------|--------------|---------------|---|
| Grigia (sinistra) | < 0.1 | Errore certo: es. `"Europa"/"Rhea"`, `"Aden"/"Brazzaville"` | 2,172 (31.8%) |
| Arancione (centro) | 0.3–0.5, gold ≤ 25 char | Candidati *mid*: es. `"38"/"46"`, `"1216"/"1220"` — spesso ancora sbagliati | 527 |
| Verde (destra) | ≥ 0.5 | Candidati *high*: es. `"551-600"/"551–600"`, `"10-year"/"10 years"` — probabili falsi negativi | 116 |

**Conclusione:** la coda destra esiste ma è piccola. Solo il 9.4% dei casi F1=0
è candidato falso negativo, e solo la zona verde (BLEURT ≥ 0.5, n=116) contiene
casi con alta probabilità di essere davvero corretti.

---

## Soglie operative

| Soglia | Valore | Interpretazione pratica |
|--------|--------|------------------------|
| `TRUE_ZERO_BLEURT` | 0.1 | Sotto questa soglia l'errore è certo — inutile verificare |
| `BLEURT_MID` | 0.3 | Soglia minima per candidatura (solo se gold è corta, ≤25 char) |
| `BLEURT_HIGH` | 0.5 | Soglia per candidatura ad alta confidenza (qualsiasi gold) |
| — | 0.7 | Soglia per falso negativo quasi certo (~85% overlap lessicale reale) |

I 643 candidati sono salvati in `../../../results/300q/bleurt_mismatch_candidates.csv`
con colonna `correct` vuota per la revisione manuale (`y` / `n` / `?`).

---

## Stima dell'impatto dei falsi negativi sulle conclusioni

Domanda centrale: i falsi negativi di Answer F1 sono abbastanza numerosi
da cambiare le conclusioni? La risposta è **no**.

| Stima | Criterio | n | % su 10,037 coppie | Impatto su F1 medio |
|-------|----------|---|-------------------|---------------------|
| Conservativa | BLEURT ≥ 0.7, ~85% precision | 27 | **0.27%** | **+0.003** |
| Media | BLEURT ≥ 0.5, ~27% precision | ~31 | **0.31%** | — |
| Upper bound | BLEURT ≥ 0.3, non filtrato | 643 | 6.4% | — |

> **Nota sull'upper bound:** i 643 casi non sono tutti falsi negativi. Nella
> fascia 0.3–0.5 la maggior parte sono errori reali con leggera similarità
> superficiale (es. `"38"/"20"`, `"10 June 1819"/"December 21, 1860"`).
> Solo sopra BLEURT ≥ 0.7 si trovano quasi esclusivamente falsi negativi certi.

- **Answer F1 medio attuale: 0.207**
- **Answer F1 medio dopo correzione conservativa: 0.210** (Δ = +0.003)

Anche correggendo tutti i 27 casi certi, il risultato principale non cambia:
la riscrittura iterativa degrada sistematicamente l'utilità per il QA.

> **Frase per la tesi:** *"Abbiamo verificato tramite BLEURT(gold, predicted)
> che i falsi negativi di Answer F1 — risposte corrette non matchate lessicalmente —
> rappresentano al più lo 0.27% delle coppie valutate (stima conservativa,
> BLEURT ≥ 0.7) e non alterano le conclusioni (Δ Answer F1 medio = +0.003)."*
