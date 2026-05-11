# Investigations aperte sul pilot 15q

_Last updated: 2026-05-09_

Questo documento traccia due investigazioni qualitative ancora aperte sul pilot 15q, che vanno al di là dei trend aggregati riportati in [data_analysis_notes.md](data_analysis_notes.md). Entrambe servono a capire **cosa succede dentro le chain**, non solo cosa dicono le medie. Per entrambe esiste già un'infrastruttura LLM-as-a-judge nel repo, da estendere o riusare.

---

## Investigazione 1 — Cosa fa davvero `elaborate`?

### Domanda

`elaborate` è l'unica istruzione il cui Answer F1 e OpenFactScore degradano in modo monotono e marcato (F1: −0.104, OFS: −0.065 da step 1 a step 3). L'ipotesi del brainstorming è che `elaborate` introduca **allucinazioni cumulative**: a ogni passo il modello aggiunge dettagli non ancorati a E₀, e questi distorcono o sostituiscono i fatti supportanti.

Da verificare:

1. Le perdite di F1 sono dovute a **fatti rimossi** o a **fatti distorti/sovrascritti** da nuovi dettagli?
2. Le aggiunte non supportate sono **innocue** (rumore che diluisce il segnale) o **conflittuali** (introducono dati errati che il QA model preferisce alla risposta corretta)?
3. C'è un pattern temporale: l'allucinazione è già presente a step 1 (eredità del primo rewrite), o si accumula a step 2 e 3?

### Stato attuale

Già pronto: [scripts/15q/elaborate_gpt_analysis.py](../../scripts/15q/elaborate_gpt_analysis.py).

Per ogni `(qid, run)` di tipo `elaborate`, lo script chiama `gpt-4o-mini` come judge sui pair consecutivi (0→1, 1→2, 2→3) e sull'overall (0→3), e produce per ogni confronto:

- `analysis`: commento libero in italiano su cosa è cambiato
- `lost`: lista di fatti scomparsi/distorti
- `added`: lista di fatti aggiunti non presenti nell'originale

Output:
- [results/15q/elaborate_gpt_analysis.csv](elaborate_gpt_analysis.csv) — strutturato (`n_lost`, `n_added`, liste)
- [results/15q/elaborate_gpt_analysis.md](elaborate_gpt_analysis.md) — report leggibile

Esempio dal report attuale (qid `2hop__14092_8311`, run 0, 0→3): la chain perde fatti come "John had five legitimate children, all by Isabella" e aggiunge claims come "King John's complex marital life saw him marry twice" — che è una distorsione, non solo elaborazione.

### Cosa manca

Il dato c'è, ma non è ancora **aggregato e correlato con F1/OFS**. Serve un secondo passaggio per:

1. Calcolare `n_lost` e `n_added` medi per step (chain elaborate, n=15 qid × 3 run = 45 chain → 45×4 confronti = 180 righe di analisi).
2. Correlare `n_added` di un confronto con il drop di Answer F1 nello stesso step. Se la correlazione è positiva, conferma il meccanismo "allucinazione → confusione del QA model".
3. **Spot-check qualitativo manuale** sulle chain dove la correlazione è più forte, per distinguere:
   - aggiunta innocua (es. "naval vessel" → "important naval vessel")
   - aggiunta conflittuale (es. inventare una data o un nome che il QA model poi riporta come risposta)
4. Un controllo specifico sulle chain dove **F1=0 a step 1 ma >0 a step 2 o 3** (vedi Investigazione 2): l'elaborate sta a volte *aggiungendo* il fatto giusto?

### Plan operativo

- Analisi dei CSV già prodotti per `elaborate_gpt_analysis` (script Python ad hoc, non serve nuova chiamata LLM).
- Output: una tabella `n_lost`/`n_added` per step × instruction (limitata a `elaborate`) e uno scatter `n_added` vs Δ-F1.
- Eventualmente estendere `elaborate_gpt_analysis.py` alle altre 3 istruzioni per confronto: una baseline su `paraphrase` chiarisce quanto le aggiunte di `elaborate` siano specifiche, dato che `paraphrase` ha F1 e OFS stabili.

---

## Investigazione 2 — Le chain con F1=0 a step 1: cosa succede dopo?

### Domanda

Su 180 chain, **96 hanno F1=0 a step 1** e sono state escluse dalle analisi aggregate (sezione 1 di [data_analysis_notes.md](data_analysis_notes.md)). L'esclusione è giusta per misurare il *drop*, ma significa che ignoriamo ~53% del dataset. Domande aperte:

1. Quante di queste chain "guariscono" più avanti (F1>0 a step 2 o 3)? Se sì, vuol dire che il primo rewrite ha rotto qualcosa che i rewrite successivi hanno casualmente ricostruito.
2. Quando il modello QA produce risposte sbagliate (non rifiuti), **da dove le prende**? Sono prese da distrattori del testo riscritto, o sono allucinazioni del QA model? Il rewriting può aver promosso un distrattore alla posizione di "fatto saliente"?
3. I rifiuti espliciti (~36) sono uniformi tra istruzioni o concentrati in `shorten`? E persistono a step 2/3 o si sbloccano?

### Stato attuale (numeri sui 720 record)

Conteggio sulle 96 chain F1=0 a step 1 (calcolato da [rewriting_chains_15q_answer_f1.csv](rewriting_chains_15q_answer_f1.csv)):

| Trajectory                       | Conteggio | Note                                                                |
|----------------------------------|-----------|---------------------------------------------------------------------|
| Stayed zero (F1=0 in tutta la chain)      | 85        | Il rewriting non ha mai trovato il fatto                            |
| Late recovery (F1>0 solo a step 2 o 3) | 8         | Il rewriting alla fine ha "ripescato" la risposta                   |
| Transient recovery (F1>0 solo a step 2)   | 3         | Recupero passeggero, poi riperso                                    |

Distribuzione per istruzione delle 11 chain con qualche recovery:

| Instruction | Late recovery | Transient recovery |
|-------------|---------------|--------------------|
| paraphrase  | 3             | 1                  |
| shorten     | 3             | 1                  |
| elaborate   | 2             | 1                  |
| formality   | 0             | 0                  |

QID coinvolti: 11 distinti. Di questi, **5 qid hanno SOLO chain con F1=0 a step 1** in tutte le istruzioni — sono domande dove il QA model sembra strutturalmente incapace di rispondere dal testo riscritto, indipendentemente dall'istruzione:

- `2hop__5658_25002`
- `2hop__61143_165532`
- `3hop1__593288_720914_41132`
- `3hop2__28727_92991_76291`
- `4hop1__9007_698949_157828_162309`

Le risposte predette nelle 96 chain (step 1) si distribuiscono in (note di `data_analysis_notes.md` §1):

- 56 risposte sbagliate (il modello dice un'altra cosa)
- 36 rifiuti espliciti ("The context does not provide…")
- 4 output vuoti (NaN)

### Cosa manca

Tre sotto-analisi qualitative:

**(a) Late recovery — perché la chain guarisce?**
Per le 11 chain "recovered", confrontare manualmente E₁, E₂, E₃ e capire cosa cambia tra lo step in cui F1=0 e lo step in cui F1>0. Tre ipotesi:

- Il rewriting successivo riesplora E₀ e ripristina un dettaglio saltato al primo passo (improbabile: il rewriting è iterativo, non rivede E₀ dopo step 1).
- Il rewriting riformula la frase in un modo che il QA model riesce a parsare meglio (sensibilità del QA allo stile, caso 3 della sezione 7 di `data_analysis_notes.md`).
- Il rewriting cambia la posizione del fatto nel testo, e questo basta al QA model.

Esempio concreto da osservare: `3hop1__305566_568433_47686 / paraphrase / run 0` — F1=0 a step 1 e 2, F1=1.0 a step 3 (gold = "Toronto Coach Terminal", pred a step 3 = "Toronto Coach Terminal", a step 1-2 = "Rodoferroviária station"). Il fatto giusto è ricomparso o il QA model ha ricominciato a vederlo?

**(b) Refusal vs wrong — separare i due failure modes**
Una funzione di classificazione automatica (regex su frasi tipo "context does not provide", "cannot answer", "not specified", …) può etichettare i 96 predicted_answer in `WRONG | REFUSAL | EMPTY`. Una volta etichettati:

- I rifiuti sono dovuti al modello QA che diventa più conservativo (caso 3 di `data_analysis_notes.md` §7) — non sono degradazione del testo. Vanno separati.
- Gli WRONG sono il caso interessante: il modello prende una risposta plausibile ma errata. Vale la pena spot-checkare 5–10 di questi per capire se la risposta è (i) presente nel testo originale come distrattore, (ii) inventata dal QA model, (iii) il fatto giusto ma normalizzato in modo diverso (es. "1216" vs "the year 1216 AD").

**(c) LLM-as-a-judge sulla correttezza semantica delle predizioni F1=0**
Il caso (iii) sopra è ben noto: la normalizzazione SQuAD-style di MuSiQue tratta come errore una risposta semanticamente corretta ma formulata diversamente. Esempio dalla §7 delle note: gold="1,400 years", pred="1400 years ago" → F1=0.8.

Soluzione: usare `gpt-4o-mini` come judge per dire se `predicted_answer` è semanticamente equivalente a `gold_answer` dato il contesto. Se sì → "F1=0 spurio", la chain è in realtà answerable e va inclusa. Se no → degradazione reale.

Questo è esattamente il pattern usato già in [scripts/15q/recall_gpt_eval.py](../../scripts/15q/recall_gpt_eval.py) (judge True/False, max_tokens=5, temperature=0). Si può adattare a questo task in poche righe — prompt di esempio:

```
Given the question, the gold answer, and the model's predicted answer,
say only YES or NO: is the prediction semantically equivalent to the gold,
ignoring formatting/wording differences?

Question: {question}
Gold: {gold_answer}
Prediction: {predicted_answer}
Equivalent?
```

Output: per ognuna delle 96 chain, una flag `gpt_judge_equivalent ∈ {YES, NO}`. Le YES vanno reincluse nelle analisi. Atteso: poche YES, ma non zero — e basta un 5-10% per cambiare la composizione del campione "answerable" da 84 a >88.

### Plan operativo

1. Script `f1_zero_diagnosis.py` (nuovo, in `scripts/15q/`) che:
   - carica `rewriting_chains_15q_answer_f1.csv`
   - filtra le 96 chain con F1=0 a step 1
   - aggiunge una colonna `category` ∈ `{WRONG, REFUSAL, EMPTY}` con regex
   - aggiunge una colonna `trajectory` ∈ `{STAYED_ZERO, LATE_RECOVERY, TRANSIENT_RECOVERY}`
   - per le righe `WRONG`, chiama gpt-4o-mini come judge (vedi sopra) e aggiunge `gpt_equivalent` ∈ `{YES, NO}`
   - salva `results/15q/rewriting_chains_15q_f1zero_diagnosis.csv`

2. Script di reporting che produce:
   - tabella refusal/wrong/empty × instruction
   - tabella trajectory × instruction
   - lista delle 11 chain "recovered" con i 4 step di testo affiancati per spot-check manuale
   - lista dei `WRONG` con `gpt_equivalent=YES` (i falsi negativi della metrica)

3. Spot-check manuale su 10 chain "stayed_zero" del tipo `WRONG`: prendono la risposta da un distrattore o la inventano? Per questo è utile vedere il testo Eₜ accanto al `predicted_answer`. Lo script può stampare il blocco rilevante.

---

## Riferimenti negli script esistenti

- `scripts/15q/elaborate_gpt_analysis.py` — judge sulla coppia (originale, rewritten) per istruzione `elaborate`. Prompt in italiano, ritorna `analysis/lost/added` come JSON. Riusa il pattern per le altre istruzioni se necessario.
- `scripts/15q/recall_gpt_eval.py` — judge minimale True/False per fact verification. È il template più semplice da clonare per il giudizio di equivalenza semantica.
- `scripts/15q/qual_check_disagreement.py` — pattern utile per stampare contesto E₀ + rewrite + facts NOT_SUPPORTED affiancati. Riusabile per lo spot-check manuale.

Tutti e tre usano `OPENAI_API_KEY` come env var, `gpt-4o-mini` con `temperature=0`, e scrivono CSV che possono essere analizzati offline.
