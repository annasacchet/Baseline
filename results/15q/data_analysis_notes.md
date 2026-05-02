# Data Analysis Notes — 15q Pilot
_Last updated: 2026-05-02_

---

## Contesto sperimentale

Questo documento riporta i risultati del pilot su 15 domande MuSiQue. L'obiettivo della tesi è rispondere a una domanda principale (**RQ1**): riscrivere iterativamente un testo ne degrada la qualità fattuale, misurata come capacità di rispondere a domande multi-hop?

### Dataset: MuSiQue

MuSiQue (Trivedi et al., 2022) è un dataset di domande multi-hop. Ogni domanda richiede di connettere 2, 3 o 4 fatti distribuiti in paragrafi separati per arrivare alla risposta. Il testo originale E₀ fornito al modello contiene 20 paragrafi: 2–4 sono i paragrafi "supporting" che contengono i fatti rilevanti, i restanti 16–18 sono distrattori.

Il pilot usa 15 domande bilanciate per hop count: 5 a 2-hop, 5 a 3-hop, 5 a 4-hop.

### Pipeline di riscrittura

Per ogni domanda, E₀ viene riscritto 3 volte con la stessa istruzione, producendo la catena E₀ → E₁ → E₂ → E₃. Le istruzioni sono tratte da OpenRewriteEval (Shu et al., 2023) e divise in due gruppi concettuali:

- **Style-oriented** (cambiano la forma, non il contenuto): `formality`, `paraphrase`
- **Content-oriented** (cambiano la struttura o la quantità di informazione): `shorten`, `elaborate`

Ogni istruzione ha 3 formulazioni diverse (run 0, 1, 2). In totale: 15 domande × 4 istruzioni × 3 run = **180 chain**, ciascuna con 4 testi (E₀, E₁, E₂, E₃) = **720 righe**.

### Metriche

**Answer F1** misura la factualità in modo indiretto: il testo riscritto Eₜ viene dato a un modello QA (OLMo-3.1-32B-Instruct), che risponde alla domanda originale. F1 token-level è calcolato tra la risposta predetta e la risposta gold.

La normalizzazione usata è quella ufficiale di MuSiQue (mutuata da SQuAD): lowercase, rimozione di punteggiatura e articoli, confronto token per token. Questa normalizzazione gestisce correttamente variazioni minori ("September 11, 1962" vs "11 September 1962" → F1=1.0) ma non copre parafrasi semantiche ("1400 years ago" vs "1,400 years" → F1=0.8, non 1.0).

**BERTScore** (roberta-large, layer 17) misura la deriva semantica del testo riscritto rispetto a E₀. Viene calcolato in due modalità:
- *Baseline*: sim(Eₜ, E₀) — quanto il testo si è allontanato dall'originale in modo cumulativo
- *Consecutive*: sim(Eₜ, Eₜ₋₁) — quanto cambia ogni singolo passo di riscrittura

---

## 1. Il problema del baseline F1 a step 0

Prima di analizzare la degradazione, è necessario capire perché non si parte da E₀ come baseline.

E₀ contiene tutti e 20 i paragrafi MuSiQue, inclusi i distrattori. Su questo testo il modello QA produce F1=0 per 8 domande su 15 — non perché i fatti manchino, ma perché:

1. Il testo è molto lungo (~2311 token) e rumoroso: il modello non riesce a localizzare i fatti rilevanti tra i distrattori.
2. Il ragionamento multi-hop è difficile: collegare 3 o 4 fatti distribuiti in paragrafi separati è un compito noto per essere difficile anche per i modelli migliori (best model ~47 F1 su MuSiQue, secondo il paper originale).

Le 8 domande con F1=0 a step 0 sono distribuite su tutti e tre i livelli di hop: 3 a 2-hop, 3 a 3-hop, 2 a 4-hop. Il fallimento non è quindi limitato alle domande più complesse — anche domande a 2-hop risultano irrispondibili da E₀, il che conferma che il problema è principalmente la lunghezza e il rumore del testo originale, non la complessità della domanda in sé.

Di conseguenza, **F1 a step 0 non è un baseline affidabile per misurare la degradazione**. Il confronto corretto è step 1 → step 3: E₁ è già una riscrittura fluida e compatta del testo, e rappresenta il punto di partenza reale dell'analisi.

### Chain answerable

Su 180 chain totali, **84 hanno F1 > 0 a step 1** (il primo testo riscritto). Queste provengono da 10 domande su 15. Solo su queste 84 chain la degradazione è misurabile in modo significativo. Tutte le analisi che seguono sono filtrate su questo sottoinsieme, salvo dove indicato diversamente.

Le 96 chain con F1=0 a step 1 si distribuiscono così:
- **56** risposte sbagliate (il modello risponde qualcosa di errato)
- **36** rifiuti espliciti ("The context does not provide...", "I cannot answer...")
- **4** output vuoti (NaN)

I rifiuti espliciti sono particolarmente importanti da notare: il modello QA, ricevendo un testo riscritto, a volte diventa più conservativo e preferisce non rispondere piuttosto che estrarre l'informazione. Questo non è degradazione fattuale del testo — è un comportamento del QA model. Separarli con `gold_in_text_check` è necessario per un'analisi precisa.

---

## 2. Answer F1 — chain answerable (n=84)

### 2.1 Trend per istruzione

| Instruction | Step 1 | Step 2 | Step 3 | Drop (t1→t3) | % chain che scendono |
|-------------|--------|--------|--------|--------------|----------------------|
| paraphrase  | 0.757  | 0.740  | 0.715  | −0.042       | 10%                  |
| formality   | 0.730  | 0.673  | 0.673  | −0.057       | 14%                  |
| elaborate   | 0.706  | 0.663  | 0.602  | −0.104       | 23%                  |
| shorten     | 0.764  | 0.599  | 0.621  | −0.143       | 25%                  |

**Osservazioni:**

`shorten` mostra il drop più netto e più rapido: F1 scende da 0.764 a step 1 a 0.599 a step 2 (−0.165 in un solo passo), poi si stabilizza leggermente a 0.621 a step 3. Il crollo a step 2 suggerisce che la compressione progressiva rimuove fisicamente i fatti chiave dal testo.

`elaborate` degrada in modo consistente ma più graduale (−0.104 da step 1 a step 3). Aggiungere dettagli non rimuove i fatti esistenti, ma può diluirli o distorcerli iterativamente.

`formality` scende da step 1 a step 2 (−0.057), poi si stabilizza. Riformulare il registro sembra sicuro dopo il primo passo.

`paraphrase` è l'istruzione più stabile (−0.042). Riformulare senza cambiare contenuto preserva quasi sempre l'informazione necessaria per rispondere.

Questo è coerente con l'ipotesi del brainstorming: le istruzioni content-oriented (`shorten`, `elaborate`) degradano di più di quelle style-oriented (`formality`, `paraphrase`).

### 2.2 Trend per hop count

| Hop | Step 1 | Step 2 | Step 3 |
|-----|--------|--------|--------|
| 2   | 1.000  | 0.920  | 0.909  |
| 3   | 0.690  | 0.591  | 0.631  |
| 4   | 0.604  | 0.552  | 0.510  |

**Osservazioni:**

Le domande 2-hop partono da F1=1.000 (tutte le chain answerable rispondono perfettamente a step 1) e degradano lentamente. Le domande 4-hop partono già più basse (0.604) e scendono monotonicamente fino a 0.510. Questo è coerente con l'ipotesi che testi con più fatti da connettere siano più vulnerabili alla riscrittura iterativa.

Le domande 3-hop mostrano un andamento non monotono (0.690 → 0.591 → 0.631): la risalita a step 3 è probabilmente rumore dato il numero piccolo di domande nel pilot (5 per hop, di cui solo 3 con chain answerable per i 3-hop).

---

## 3. BERTScore — deriva semantica (chain answerable)

### 3.1 Baseline (vs E₀) per istruzione

| Instruction | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|
| formality   | 0.918  | 0.903  | 0.897  |
| elaborate   | 0.910  | 0.889  | 0.875  |
| paraphrase  | 0.882  | 0.865  | 0.861  |
| shorten     | 0.842  | 0.825  | 0.818  |

**Osservazioni:**

BERTScore scende monotonicamente per tutte le istruzioni — ogni passo di riscrittura allontana il testo da E₀ in modo cumulativo. Questo è il segnale più pulito e consistente nell'intero dataset: non è influenzato dai fallimenti del QA model.

L'ordinamento è stabile in tutti i passi: `formality` causa meno deriva semantica, `shorten` la massima. Questo riflette direttamente la perdita di contenuto: un testo compresso a 464 token (da 2311) è inevitabilmente molto diverso dall'originale in embedding space.

### 3.2 Consecutivo (vs Eₜ₋₁) — overall

| Step | BERTScore consecutivo |
|------|----------------------|
| 1    | 0.889                |
| 2    | 0.941                |
| 3    | 0.960                |

**Osservazioni:**

Ogni singolo passo introduce cambiamenti sempre più piccoli (la similarità consecutiva sale verso 1.0), ma la deriva cumulativa dalla baseline continua ad aumentare. Questo conferma che la degradazione è un fenomeno **cumulativo**: ogni riscrittura sembra conservativa presa singolarmente, ma l'effetto si accumula attraverso le iterazioni.

---

## 4. Token count

| Instruction | Step 0 | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|--------|
| elaborate   | 2311   | 1506   | 1543   | 1638   |
| formality   | 2311   | 1310   | 1229   | 1197   |
| paraphrase  | 2311   | 926    | 830    | 788    |
| shorten     | 2311   | 603    | 505    | 464    |

**Osservazioni:**

Tutte le istruzioni comprimono drasticamente il testo da E₀ (2311 token) al primo passo. Questo è un artefatto strutturale: E₀ contiene 20 paragrafi e il modello produce una riscrittura coerente e compatta. `elaborate` è l'unica istruzione che aumenta progressivamente la lunghezza da step 1 in poi (1506 → 1638), come atteso.

La forte compressione da E₀ a E₁ spiega perché F1 non si può misurare come drop da step 0: il testo cambia radicalmente già al primo passo per ragioni strutturali, non solo per perdita di fatti.

---

## 5. Relazione tra F1 e BERTScore

Le due metriche misurano cose diverse e non sempre concordano:

- **BERTScore** misura quanto il testo è cambiato semanticamente da E₀. Scende sempre e per tutte le istruzioni.
- **Answer F1** misura se il fatto specifico necessario per rispondere è ancora presente e recuperabile. È molto più rumoroso.

È possibile avere BERTScore basso (testo molto cambiato) ma F1 alto, se il fatto chiave è ancora presente. È possibile l'inverso: BERTScore alto (testo poco cambiato) ma F1=0, se il modello QA fallisce il ragionamento multi-hop o produce un rifiuto esplicito.

La combinazione più informativa è: BERTScore conferma che la deriva semantica c'è sempre e cresce con le iterazioni; F1 condizionato alle chain answerable mostra che la degradazione fattuale è reale ma variabile per istruzione.

---

## 6. OpenFactScore — fedeltà fattuale rispetto a E₀

OpenFactScore (OFS) misura la fedeltà fattuale in modo diretto e indipendente dal modello QA: il testo riscritto Eₜ viene decomposto in fatti atomici (AFG: OLMo-2-1124-7B-SFT), ciascuno verificato contro E₀ come knowledge source (AFV: Gemma-3-4b-it). Il risultato è la proporzione di fatti supportati da E₀.

A differenza di Answer F1, OFS non richiede che il testo sia "answerable" — misura quanti fatti del testo riscritto sono ancora ancorati all'originale, indipendentemente dalla capacità del QA model di estrarre la risposta.

OFS è calcolato su **tutte le 180 chain** (540 righe = 180 chain × 3 step), filtrate poi sulle 84 chain answerable per consistenza con le altre metriche.

### 6.1 FactScore per istruzione (84 chain answerable)

| Instruction | Step 1 | Step 2 | Step 3 | Drop (t1→t3) |
|-------------|--------|--------|--------|--------------|
| paraphrase  | 0.909  | 0.907  | 0.909  | −0.000       |
| formality   | 0.899  | 0.903  | 0.899  | −0.000       |
| shorten     | 0.902  | 0.893  | 0.894  | −0.008       |
| elaborate   | 0.872  | 0.846  | 0.807  | −0.065       |

**Osservazioni:**

`elaborate` è l'istruzione più dannosa per la fedeltà fattuale: scende da 0.872 a 0.807, con il calo che accelera a step 3. Aggiungere dettagli introduce progressivamente fatti non supportati da E₀ (allucinazioni rispetto alla fonte).

`paraphrase` e `formality` sono stabili — il FactScore non scende. Riformulare stile non introduce nuovi fatti.

`shorten` ha un drop molto piccolo (−0.008): la compressione non *introduce* fatti sbagliati, ma *rimuove* quelli necessari per rispondere. Questo spiega la divergenza con Answer F1 (`shorten` ha il drop F1 maggiore −0.143 ma il drop OFS minore): le due metriche misurano aspetti complementari della degradazione.

### 6.2 FactScore per hop count (84 chain answerable)

| Hop | Step 1 | Step 2 | Step 3 | Drop (t1→t3) |
|-----|--------|--------|--------|--------------|
| 2   | 0.915  | 0.897  | 0.889  | −0.026       |
| 3   | 0.888  | 0.886  | 0.859  | −0.029       |
| 4   | 0.862  | 0.851  | 0.832  | −0.030       |

**Osservazioni:**

Il gradiente 2-hop > 3-hop > 4-hop è presente e monotono per tutti gli step: le domande più complesse hanno testi che partono già con FactScore più basso e degradano di più. È coerente con Answer F1 e con l'ipotesi della tesi.

### 6.3 Relazione OFS — Answer F1

Le due metriche misurano aspetti complementari:

- **OFS** misura quanti fatti del testo riscritto sono supportati da E₀. È sensibile all'*introduzione* di fatti errati (allucinazioni).
- **Answer F1** misura se il fatto specifico per rispondere è ancora presente e recuperabile. È sensibile alla *rimozione* di fatti.

`elaborate`: OFS scende molto (allucinazioni), Answer F1 scende moderatamente.
`shorten`: OFS quasi stabile (non introduce errori), Answer F1 scende molto (rimuove fatti).

Questa dissociazione è uno dei risultati più interessanti del pilot.

---

## 7. Limiti dell'analisi attuale

**Dimensione del pilot.** 15 domande, di cui 10 con chain answerable. I pattern per hop count (specialmente 3-hop) sono instabili. I risultati sono indicativi, non conclusivi.

**Normalizzazione Answer F1.** La normalizzazione MuSiQue (SQuAD-style) gestisce bene variazioni ortografiche e di articoli, ma non parafrastica. Una risposta semanticamente corretta ma formulata diversamente può ricevere F1 basso. Il `gold_in_text_check` aiuta a separare questi casi, ma non è ancora integrato sistematicamente nell'analisi.

**QA model e rifiuti.** Il `gold_in_text_check` (analisi su 720 righe) classifica ogni riga in 4 categorie:

| Categoria | N | Significato |
|-----------|---|-------------|
| hit | 228 | Gold presente nel testo, F1>0 — risposta corretta |
| degraded | 216 | Gold assente dal testo, F1=0 — **vera degradazione fattuale** |
| false_negative | 172 | Gold presente ma F1=0 — **errore del QA model**, non del testo |
| parametric_memory | 104 | Gold assente ma F1>0 — il modello risponde dalla memoria interna |

172 false negatives significa che in molti casi il fatto era ancora nel testo riscritto ma il QA model non è riuscito a estrarlo — rifiuto esplicito o risposta sbagliata nonostante il gold fosse presente. Questi non sono degradazione fattuale.

La conseguenza più importante emerge filtrando solo le chain answerable (84) e condizionando al gold presente nel testo:

| | Step 1 | Step 2 | Step 3 | Drop |
|--|--------|--------|--------|------|
| F1 grezzo (answerable) | 0.738 | 0.669 | 0.652 | −0.086 |
| F1 condizionato (gold presente) | 0.817 | 0.764 | 0.760 | −0.057 |

Quando il gold è fisicamente presente nel testo riscritto, il F1 parte più alto (0.817) e degrada molto meno (−0.057 vs −0.086). La differenza tra i due — circa 0.03 punti di drop — è il contributo del QA model che fallisce su testi trasformati anche quando il fatto è ancora lì.

Questo implica che **la vera degradazione fattuale è più lenta di quanto appaia dal F1 grezzo**. Il grafico `traj_f1_gold_in_text.pdf` mostra i due pannelli affiancati per istruzione.

**OpenFactScore.** Completato su tutte le 15 domande (540 righe). Vedi sezione 6.

**PAU analysis.** Non ancora eseguita (richiede n=5 ripetizioni per chain). Permetterà di distinguere capability erosion, stability loss e combined degradation (Laban et al., 2024).

**Self-Refine (RQ3).** Pipeline pronto (`self_refine_pipeline.py`). Smoke test completato (1 domanda, temperature=0.7). Run completo in corso su Homer con temperature=0.0 (stessa del rewriting baseline) per garantire confronto diretto.

---

## 8. Grafici disponibili

Tutti i grafici sono filtrati sulle 84 chain answerable (F1 > 0 a step 1), salvo `degradation_f1.pdf`.

| File | Contenuto |
|------|-----------|
| `traj_f1_by_instruction.pdf` | F1 step 1→3, media ± std per istruzione |
| `traj_bert_by_instruction.pdf` | BERTScore step 1→3, media ± std per istruzione |
| `traj_f1_by_hop.pdf` | F1 step 1→3, una linea per hop count |
| `traj_f1_vs_bert.pdf` | F1 e BERTScore affiancati, una linea per istruzione |
| `traj_f1_heatmap.pdf` | F1 per domanda × step, ordinato per hop count |
| `traj_f1_dotplot.pdf` | F1 per domanda, traiettorie individuali step 1→3 |
| `traj_style_vs_content.pdf` | F1 e BERTScore: style-oriented vs content-oriented |
| `traj_factscore_by_hop.pdf` | OpenFactScore step 1→3, una linea per hop count |
| `traj_f1_gold_in_text.pdf` | F1 grezzo vs F1 condizionato al gold presente, per istruzione |
| `chain_status_by_hop.pdf` | Distribuzione hit/degraded/refusal/empty per hop count (step 1) |
| `degradation_f1.pdf` | Confronto tutte le chain vs solo answerable (pannelli affiancati) |
