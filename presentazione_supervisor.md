# Factuality Degradation in Iterative LLM Rewriting — Baseline Report

**Anna Sacchet — Master's Thesis**
Presentation for supervisors — 2026-04-24

---

## 1. Posizionamento

Questo documento riassume lo stato del lavoro a seguito del brainstorming del 21 aprile 2026 ([brainstormingING.md](brainstormingING.md)). Il brainstorming fissava tre research question (degradazione, effetto dell'istruzione/hop, Self-Refine) e il disegno sperimentale; qui riporto il **primo test baseline effettivamente eseguito end-to-end**, con pipeline, metriche, limitazioni attuali e prime osservazioni.

Lo scopo è duplice:
1. validare che la pipeline tecnica (rewriting → estrazione atomic facts → verifica → BERTScore → token counts) giri su dati reali;
2. produrre una prima lettura dei risultati su cui discutere scelte e prossimi passi.

---

## 2. Cosa è stato fatto fin ora

### 2.1 Dataset e campione pilota

- **Dataset**: MuSiQue-Ans (`musique_ans_v1.0_dev.jsonl`), coerente con il brainstorming.
- **Campione del baseline attuale**: **1 sola question `qid = 2hop__635544_110949`**
  ("What is the date of birth of the person who was part of Ratata?"), categoria **2-hop**.
- **Testo E₀ (step 0)**: i paragrafi di evidenza concatenati (≈ 2357 token del tokenizer del modello).
- **Condizioni testate**:
  - 2 gruppi × 2 istruzioni = 4 istruzioni totali
    - *content*: `elaborate`, `shorten`
    - *style*: `formality`, `paraphrase`
  - 3 step di riscrittura (t₁, t₂, t₃) oltre a t₀
  - **3 run** per condizione (per i token counts) → per FactScore/BERTScore è attualmente disponibile solo **run 0**.

Questo è quindi un **pilot su 1 question, 2-hop**: sufficiente per verificare la pipeline, **non** per trarre conclusioni statisticamente valide sulle RQ.

### 2.2 Pipeline implementata

File principali ([scripts/](scripts/)):

1. [export_chain_steps.py](scripts/export_chain_steps.py) — estrae gli step dalle chain prodotte dal modello rewriter.
2. [factscore_eval.py](scripts/factscore_eval.py) — calcola il FactScore variante "source-faithfulness".
3. [add_bertscore.py](scripts/add_bertscore.py) — aggiunge BERTScore Precision/Recall/F1.
4. [visualize_factscore_bertscore.py](scripts/visualize_factscore_bertscore.py) — genera `analisi_factscore_bertscore.pdf`.
5. [visualize_per_step.py](scripts/visualize_per_step.py) — genera `evoluzione_per_step.pdf`.
6. [visualize_token_counts.py](scripts/visualize_token_counts.py) — genera i PDF `token_counts_*.pdf`.

### 2.3 Modelli usati

| Ruolo | Modello | Note |
|---|---|---|
| **Rewriter** (produce E₁, E₂, E₃) | OLMo-3 32B (famiglia del brainstorming) | Modello target della tesi |
| **Judge** (estrazione atomic facts + verifica SUPPORTED / NOT_SUPPORTED / CONTRADICTED) | **`gpt-4o-mini` via API OpenAI** | Variante rispetto al setup canonico — vedi limitazioni |
| **Embedder BERTScore** | `roberta-large`, layer 17, lingua `en` | Setting standard della libreria `bert_score` |

---

## 3. Metriche: cosa misuro e *in cosa mi discosto dalle versioni canoniche*

Questa sezione è importante per i supervisor: **sto usando FactScore e BERTScore in un modo non canonico**. Non è un errore, ma va dichiarato esplicitamente.

### 3.1 FactScore — variante source-faithfulness

**Canonical FactScore** (Min et al., 2023):
- estrae claim atomiche dal testo generato;
- ogni claim viene verificata contro **Wikipedia** (retrieval + entailment), su biografie di entità reali;
- pensato per misurare l'accuratezza *enciclopedica* di un testo generato da un LLM.

**La mia variante** (vedi [factscore_eval.py:1-8](scripts/factscore_eval.py#L1-L8)):
- estrazione delle atomic facts **canonica**: sentence tokenization + BM25 few-shot con `demons.json` ufficiale di Min et al. (7 fixed demos + 1 BM25 demo);
- verifica delle claim **non** contro Wikipedia, ma contro **E₀**, cioè il testo originale di step 0 della stessa chain;
- il judge (`gpt-4o-mini`) riceve E₀ come contesto e decide se la claim è SUPPORTED / NOT_SUPPORTED / CONTRADICTED;
- penalità di lunghezza: `min(1, n_facts / γ)` con γ = 10 (come nel paper), che non morde sui nostri testi (sempre ≥ 38 facts).

**Perché ha senso nel nostro setup.** La research question non è "il modello sa la verità del mondo?", ma "il modello preserva l'informazione del testo che gli abbiamo dato?". La ground truth rilevante è quindi E₀, non Wikipedia. FactScore-rispetto-a-E₀ è quello che in letteratura si chiama *faithfulness* o *source-grounded factuality*.

**Limitazione nota.** Un fatto può essere vero nel mondo ma assente da E₀: viene conteggiato come NOT_SUPPORTED. Questo è voluto (stiamo misurando conservazione di E₀) ma **da dichiarare** quando si interpreta un calo di FactScore — parte del calo è "il modello aggiunge conoscenza esterna", non necessariamente hallucination.

### 3.2 BERTScore — usato tra E₀ e Eₜ

**Canonical BERTScore** (Zhang et al., 2020): similarità semantica token-level tra un *candidate* e un *reference*, tipicamente su coppie traduzione/summary dove esiste un reference gold.

**Come lo sto usando**:
- reference = E₀ (il testo originale di partenza);
- candidate = Eₜ (lo step t della stessa chain);
- riporto Precision, Recall, F1.

**Differenza rispetto al suo uso canonico**:
- qui non c'è un reference "gold" in senso stretto: E₀ è il punto di partenza, non una traduzione di riferimento;
- BERTScore misura quindi **drift semantico cumulativo rispetto a E₀**, non qualità assoluta di un output;
- attenzione all'interpretazione: un BERTScore F1 basso su `shorten` non significa "riscrittura brutta", significa "testo semanticamente lontano da E₀" — che per `shorten` è inevitabile e in parte desiderato.

Per questo motivo nel brainstorming BERTScore è classificato come **metrica secondaria** (drift), non primaria (factuality).

### 3.3 Token counts — tokenizer del modello

**Scelta importante**: i conteggi di token non sono fatti con un tokenizer generico (es. `tiktoken`), ma **con il tokenizer del modello rewriter stesso** (OLMo-3). Questo ha due implicazioni:

- i valori sono direttamente interpretabili come "carico di generazione" del modello, non come conteggi word-level;
- confronti cross-model non sarebbero 1:1 — se in futuro si aggiunge LLaMA o Mistral, i valori in token **non sono comparabili direttamente** e conviene normalizzare (es. token/word ratio, o riportare anche char-level length).

---

## 4. Risultati del baseline (1 question, 2-hop)

Dati da [results/rewriting_chains32b_factscore_bertscore.csv](results/rewriting_chains32b_factscore_bertscore.csv) e [results/rewriting_chains32b_token_counts.csv](results/rewriting_chains32b_token_counts.csv).

### 4.1 Tabella di sintesi (run 0)

| Istruzione | Gruppo | FS@1 | FS@2 | FS@3 | BS@1 | BS@2 | BS@3 | Δ FS (1→3) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| elaborate  | content | 0.908 | 0.893 | 0.882 | 0.901 | 0.883 | 0.869 | −0.026 |
| shorten    | content | 0.904 | 0.886 | 0.842 | 0.802 | 0.790 | 0.776 | −0.062 |
| formality  | style   | 0.922 | 0.917 | 0.896 | 0.942 | 0.920 | 0.884 | −0.026 |
| paraphrase | style   | 0.935 | 0.904 | 0.850 | 0.911 | 0.842 | 0.834 | −0.085 |

(FS = FactScore, BS = BERTScore F1, @k = step k)

### 4.2 Grafico 1 — Evoluzione per step

**File**: [results/evoluzione_per_step.pdf](results/evoluzione_per_step.pdf)

*Cosa mostra*: due pannelli affiancati. A sinistra FactScore per step (t₁ → t₃), a destra BERTScore F1 per step. Una linea per istruzione (4 linee).

*Come leggerlo in presentazione*:
> "Tutte e quattro le istruzioni mostrano un trend monotono decrescente, sia su FactScore sia su BERTScore, passando da t₁ a t₃. Questo è il primo indizio a supporto di RQ1: **la riscrittura iterativa degrada la factuality anche in una singola question 2-hop**. Non c'è auto-stabilizzazione, il segnale non è piatto."

### 4.3 Grafico 2 — Analisi combinata FactScore × BERTScore (4 pannelli)

**File**: [results/analisi_factscore_bertscore.pdf](results/analisi_factscore_bertscore.pdf)

*Cosa mostra* (pagina 1):
- **scatter FS vs BS** con Pearson r = **0.712**;
- evoluzione FS e BS sovrapposte per istruzione;
- boxplot della distribuzione di FS e BS per istruzione;
- stacked bar della composizione dei fatti (SUPPORTED / NOT_SUPPORTED / CONTRADICTED).

*Come leggerlo*:
> "La correlazione FS-BS è moderata-forte (r ≈ 0.71), ma **non è 1**: ci sono condizioni in cui i due segnali divergono. Esempio chiaro: `shorten` ha FS relativamente alto (≈ 0.88 in media) ma BS ≈ 0.79 — il riassunto conserva i fatti che cita, ma semanticamente si allontana molto da E₀ perché ne cita pochi. Questo è proprio il tipo di informazione che **una sola metrica non dà**: ecco perché le usiamo insieme."

Punto chiave per i supervisor: la divergenza `shorten`-bassa-BS, `shorten`-alto-FS è **attesa per costruzione di FactScore** (score calcolato sui fatti *che ci sono*, non su quelli *che sono spariti*). È un argomento forte per affiancare Answer F1 nei prossimi round: Answer F1 penalizza la perdita del fatto specifico necessario per rispondere, cosa che FactScore in questa forma non cattura.

### 4.4 Grafico 3 — Token counts per istruzione (per-group, multi-run)

**Files**:
- [results/token_counts_content.pdf](results/token_counts_content.pdf) — `elaborate` e `shorten` su 3 run
- [results/token_counts_style.pdf](results/token_counts_style.pdf) — `formality` e `paraphrase` su 3 run
- [results/token_counts_comparison.pdf](results/token_counts_comparison.pdf) — content vs style a confronto

*Valori medi su 3 run* (tokenizer del modello):

| Istruzione | t₀ | t₁ | t₂ | t₃ |
|---|---:|---:|---:|---:|
| elaborate  | 2357 | 1721 | 1667 | 1619 |
| shorten    | 2357 |  683 |  572 |  437 |
| formality  | 2357 | 1267 | 1199 | 1138 |
| paraphrase | 2357 | 1129 |  907 |  875 |

*Come leggerlo*:
> "Nessuna istruzione produce un testo più lungo di E₀, **inclusa `elaborate`**. Questo è inatteso: `elaborate` dovrebbe espandere. Due ipotesi:
> 1. il prompt di `elaborate` funziona più come 'rielabora' che come 'espandi';
> 2. il modello, dovendo elaborare testo lungo e multi-paragrafo, tende comunque a compattarlo.
>
> `shorten` compatta aggressivamente (quasi −81% al t₃). `paraphrase` e `formality` hanno un comportamento intermedio. La cosa importante: **la lunghezza non spiega da sola il calo di FactScore** — `formality` cala in FS pur mantenendo più testo di `paraphrase`."

### 4.5 Grafico 4 — Evoluzione token counts e varianza

**Files**:
- [results/evoluzione_token_counts.pdf](results/evoluzione_token_counts.pdf)
- [results/variazione_token_counts.pdf](results/variazione_token_counts.pdf)

*Cosa mostrano*: traiettorie di lunghezza nei 3 run per ogni istruzione, utile per vedere quanta variabilità c'è tra run (precursore della futura analisi P/A/U).

*Come leggerlo*:
> "Già su 3 run vediamo che `shorten` ha varianza elevata in t₁ (range 373–918 token): il modello è **instabile** su come comprime. `elaborate` e le style hanno varianza minore. Questa è una prima indicazione che la **Unreliability** del framework P/A/U sarà informativa, non solo rumore."

---

## 5. Limitazioni attuali — da esplicitare in presentazione

Queste sono le cose che i supervisor **devono sapere** prima di interpretare i numeri.

### 5.1 Campione

- **1 sola question**, 2-hop. Nessuna conclusione su RQ2 (effetto hop count) è possibile finora.
- Solo `run 0` per FactScore/BERTScore. P/A/U richiede ≥ 5 run (brainstorming §5.1). Attualmente non ho le distribuzioni necessarie per calcolare i tre statistics (P, A, U).
- Self-Refine (RQ3) non è ancora stato implementato — il pipeline attuale è la baseline senza refine.

### 5.2 Scelta del judge: API OpenAI (`gpt-4o-mini`)

- Uso **API OpenAI** sia per estrazione atomic facts sia per verifica. Costi contenuti ma **non trascurabili** quando scalerò a 405 question × 4 istruzioni × 3 step × ≥ 5 run.
- `gpt-4o-mini` è un modello **chiuso e proprietario**, il che significa:
  - non riproducibilità bit-a-bit (il modello può essere aggiornato lato OpenAI);
  - dipendenza da un servizio esterno (rate limit, downtime, cambiamenti di pricing);
  - potenziale conflitto metodologico: uso un modello proprietario per giudicare l'output di modelli open (OLMo). Da discutere se per la versione finale convenga un judge open-weights (es. Llama-3.1-70B o un modello NLI dedicato) per avere il setup completamente riproducibile.
- Inoltre, il prompt di verifica accetta paraphrase-come-SUPPORTED: questo è desiderabile per non falsare i risultati, ma introduce una dipendenza dalla qualità linguistica di `gpt-4o-mini`.

### 5.3 FactScore non-canonico

- Come scritto in §3.1, sto usando FactScore rispetto a E₀ invece che rispetto a Wikipedia. Il nome "FactScore" quindi va sempre qualificato — in tesi sarà meglio riferirvisi come *source-grounded FactScore* o *faithfulness score* per evitare confusione con il paper originale.

### 5.4 Answer F1 non ancora calcolato

- Il brainstorming lo classifica come metrica primaria insieme a FactScore. Nel baseline attuale **non è stato ancora calcolato**. È il tassello più importante da aggiungere prima di scalare l'esperimento, perché è l'unico che penalizza direttamente la perdita del fatto critico per rispondere alla question.

### 5.5 Hardcoded paths e TEST_MODE

- Gli script contengono path assoluti (`/Users/annasacchet/Desktop/...`) e `TEST_MODE = True` di default in [factscore_eval.py:36](scripts/factscore_eval.py#L36). Non è un problema scientifico ma è debito tecnico da ripulire prima di far girare l'esperimento completo.

---

## 6. Prime conclusioni (con cautela)

Con il caveat forte che siamo su **1 question**:

1. **RQ1 — la degradazione c'è ed è monotona.** Tutte e quattro le istruzioni mostrano un calo di FactScore da t₁ a t₃ (−0.026 a −0.085). Coerente con l'ipotesi del brainstorming. Nessuna evidenza di auto-stabilizzazione.

2. **RQ2 — segnale preliminare: paraphrase degrada tanto quanto content.** Inatteso rispetto all'ipotesi ("style < content"): `paraphrase` (style) ha il Δ FactScore maggiore (−0.085), superiore a `shorten` (−0.062) e a `elaborate` (−0.026). Se confermato su più question, vorrebbe dire che **la distinzione style/content non è predittiva**, o che `paraphrase` è in realtà più "content-heavy" di quanto sembri (la riformulazione lessicale rimpiazza attivamente termini e può perdere entità).
   - Counter-ipotesi alternativa: `paraphrase` produce testi *molto più corti* (1129 → 875 token) pur essendo style, quindi parte del calo di FS potrebbe essere omissione — esattamente ciò che Answer F1 servirebbe a misurare.

3. **FactScore e BERTScore sono complementari, non sostituibili.** r ≈ 0.71 significa che circa metà della varianza non è condivisa. L'esempio `shorten` (FS alto, BS basso) è il caso più chiaro del perché servono entrambi.

4. **Le lunghezze rivelano comportamenti inaspettati.** `elaborate` non elabora (non allunga). Da rivedere il prompt di `elaborate` o accettare che in regime iterativo anche istruzioni di espansione portano a contrazione.

5. **La variabilità tra run esiste già a 3 run** (soprattutto per `shorten`). Questo legittima l'approccio P/A/U a ≥ 5 run previsto nel brainstorming.

---

## 7. Cosa propongo per la prossima milestone

Prima di scalare a 405 question × ogni condizione, prioritizzerei:

1. **Aggiungere Answer F1** alla pipeline — è la metrica primaria che manca, e le prime conclusioni su `paraphrase`/`shorten` si reggono o cadono su quella.
2. **Girare ≥ 5 run** sulla stessa 1 question, così da poter calcolare P/A/U già sul pilot e vedere se l'infrastruttura regge.
3. **Espandere a ~5 question per hop-count** (2/3/4-hop) come smoke test prima dello scale-up completo.
4. **Decidere sul judge**: restiamo su `gpt-4o-mini` (costo basso, veloce, ma proprietario) o passiamo a un judge open-weights? Questa decisione va presa *prima* dello scale-up, perché cambiarla dopo invalida i risultati precedenti.
5. **Implementare Self-Refine** (RQ3) solo dopo che il baseline è stabile, non prima.

---

*Fine documento.*
