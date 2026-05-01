# Data Analysis Notes — 15q Pilot
_Last updated: 2026-05-01_

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

**Answer F1** misura la factualità in modo indiretto: il testo riscritto Eₜ viene dato a un modello QA (OLMo-2-1124-32B-Instruct), che risponde alla domanda originale. F1 token-level è calcolato tra la risposta predetta e la risposta gold.

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

## 6. Limiti dell'analisi attuale

**Dimensione del pilot.** 15 domande, di cui 10 con chain answerable. I pattern per hop count (specialmente 3-hop) sono instabili. I risultati sono indicativi, non conclusivi.

**Normalizzazione Answer F1.** La normalizzazione MuSiQue (SQuAD-style) gestisce bene variazioni ortografiche e di articoli, ma non parafrastica. Una risposta semanticamente corretta ma formulata diversamente può ricevere F1 basso. Il `gold_in_text_check` aiuta a separare questi casi, ma non è ancora integrato sistematicamente nell'analisi.

**QA model e rifiuti.** Il 37.5% delle chain con F1=0 a step 1 sono rifiuti espliciti del modello QA ("cannot answer from context"), non degradazione fattuale del testo riscritto. Questo gonfia il numero di chain "non answerable" e richiede un'analisi separata.

**OpenFactScore.** Attualmente disponibile solo per 1 domanda su 15 (pilot). Quando completato su tutte le 15 domande, fornirà un segnale più diretto sulla perdita di fatti atomici, indipendente dal QA model.

**PAU analysis.** Non ancora eseguita (richiede n=5 ripetizioni per chain). Permetterà di distinguere capability erosion, stability loss e combined degradation (Laban et al., 2024).

**Self-Refine (RQ3).** Non ancora eseguito. Il pipeline è pronto (`self_refine_pipeline.py`).

---

## 7. Grafici disponibili

Tutti i grafici sono filtrati sulle 84 chain answerable (F1 > 0 a step 1).

| File | Contenuto |
|------|-----------|
| `traj_f1_by_instruction.pdf` | F1 step 1→3, media ± std per istruzione |
| `traj_bert_by_instruction.pdf` | BERTScore step 1→3, media ± std per istruzione |
| `traj_f1_by_hop.pdf` | F1 step 1→3, una linea per hop count |
| `traj_f1_vs_bert.pdf` | Traiettorie (BERTScore, F1) per chain, griglia per istruzione |
| `traj_f1_heatmap.pdf` | F1 per domanda × step, una colonna per istruzione |
| `degradation_f1.pdf` | Confronto tutte le chain vs solo answerable (pannelli affiancati) |
