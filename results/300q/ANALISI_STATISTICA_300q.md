# Analisi statistica 300q — Rewriting iterativo MuSiQue (OLMo-3.1-32B-Instruct, 4-bit)

> **Disegno:** 297 qid × 4 istruzioni × 3 run × 4 step = 14 256 osservazioni F1; 10 692 per OFS / BERTScore / BLEURT.
> **Pipeline:** [scripts/300q/statistical_analysis_300q.py](../../scripts/300q/statistical_analysis_300q.py) · **Output:** [stats_v2/](stats_v2/) · **Bootstrap B = 10 000**

---

## Tabella di lettura: RQ → finding → evidenza

| RQ | Domanda | Conclusione | Stima chiave (95% CI) | Test |
|----|---------|-------------|------------------------|------|
| **RQ1** | Il primo step causa un crollo di F1? | **Sì**, collasso concentrato 0→1. | ΔF1 = −0.147 [−0.190, −0.103] | Wilcoxon paired Holm, p < 10⁻⁷; rank-biserial = −0.43 |
| **RQ1b** | Gli step successivi sono equivalenti a zero? | **No** per 1→2; **sì** al pelo per 2→3. | Δ1→2 = −0.025 [−0.034, −0.016]; Δ2→3 = −0.014 [−0.021, −0.006] | TOST ±0.02: p = 0.85 / 0.05 |
| **RQ2** | La fattualità decade progressivamente? | **Sì**, ~1 pp/step con effect size grande. | ΔOFS 1→3 = −0.025 [−0.029, −0.022] | Wilcoxon paired, rank-biserial −0.62 / −0.56 |
| **RQ3** | La lunghezza media il crollo di F1? | **Sì**, ~48% dell'effetto step→F1 passa via log(n_tokens). | indirect = −0.028 [−0.039, −0.017]; prop = 0.48 [0.31, 0.66] | Mediazione causale (Baron–Kenny), cluster bootstrap |
| **RQ4** | Il recovery apparente è stabile? | **No**, rumore stocastico. | 40% dei recuperi perso dopo 1 step; 11.2% [8.2%, 14.5%] @F1≥0.9 | Kaplan–Meier + log-rank istruzione (p = 0.46) |
| **RQ4b** | Quanto è sensibile il recovery alla soglia? | **Molto**: 22% @qualsiasi overlap → 11% @F1≥0.9. | Vedi §7 (sensitivity curve) | Cluster bootstrap su qid |

---

## Sommario metodologico

### Unità di analisi
- **Unità sperimentale:** la singola `qid` (297 domande MuSiQue). Le 12 catene per qid (4 istruzioni × 3 run) sono replicati interni.
- **Aggregazione per test paired:** media F1 / OFS / BERTScore / BLEURT entro qid, poi confronto step-by-step su 297 valori. Questa scelta evita di gonfiare i p-value per pseudo-replicazione.

### Modelli statistici
1. **Friedman omnibus**: test non parametrico ranks-based, prerequisito per i contrasti pianificati.
2. **Wilcoxon paired step-by-step**, correzione **Holm** per il numero di contrasti.
3. **Effect size rank-biserial** (matched pairs): scala 0–1, interpretabile come "frazione di pairs in cui il secondo step è peggio del primo".
4. **Mixed models** (`statsmodels.mixedlm`): random intercept per qid; outcome continui → LMM, F1 binario → linear-probability model con random intercept (riportato come check, non come inferenza primaria).
5. **ICC(qid)**: var(RE) / [var(RE) + var(residual)] dal modello null. Misura quanto la varianza è "dentro vs tra qid".
6. **Cluster bootstrap** su qid, B = 10 000, percentile-CI al 95%. Vettorizzato: ~5 µs per iterazione (vs 30 ms naïve).
7. **Mediazione causale** alla Baron–Kenny:
   - c = effetto totale (outcome ~ step + covariates)
   - a = step → mediatore (log(n_tokens))
   - b = mediatore → outcome, controllando step
   - c' = effetto diretto
   - ab = c − c' = effetto indiretto (mediato dalla lunghezza)
   - prop_mediated = ab / c, con CI bootstrap a livello qid.
8. **Survival analysis** sui recuperi: per ogni catena zero-start con F1 ≥ 0.9 in almeno uno step, `duration` = numero di step consecutivi di "sopravvivenza" della risposta corretta; `event=1` se persa entro step 3, `event=0` se censored al limite.
9. **TOST equivalence** (Two One-Sided Tests, ±0.02 F1): permette di concludere *equivalenza* a zero invece del solo "n.s.".
10. **Sensitivity analysis**: tasso di recovery come funzione della soglia F1, con CI cluster-bootstrap.

### Caveat
- I p-value di Friedman e Wilcoxon possono comparire come `0.000000` nei CSV: significa "sotto il floor numerico", non zero esatto. Il valore numerico esatto è ricavabile dal `chi2`/`statistic`.
- La mediazione assume linearità nelle equazioni di path. Su F1 (bimodale) abbiamo verificato che i risultati restano stabili anche modellando F1 come binario (LPM): la prop_mediated non cambia di più di 2 pp.

---

## Risultati principali

### S0 — Descrittive per step
[stats_v2/s0_descriptive_by_step.csv](stats_v2/s0_descriptive_by_step.csv)

| Metrica | step 0 | step 1 | step 2 | step 3 |
|---------|--------|--------|--------|--------|
| Answer F1 | 0.362 | 0.215 | 0.191 | 0.177 |
| OFS | — | 0.863 | 0.849 | 0.838 |
| BERTScore baseline | — | 0.853 | 0.845 | 0.841 |
| BERTScore consecutive | — | 0.853 | 0.938 | 0.952 |
| BLEURT baseline | — | 0.398 | 0.371 | 0.358 |
| BLEURT consecutive | — | 0.398 | 0.637 | 0.677 |
| BLEURT answer | — | 0.304 | 0.289 | 0.281 |
| n_tokens (media) | 2 445 | 665 | 535 | 479 |

### S1 — Friedman + Wilcoxon paired
[stats_v2/s1_friedman_omnibus.csv](stats_v2/s1_friedman_omnibus.csv) · [stats_v2/s1_wilcoxon_step_contrasts.csv](stats_v2/s1_wilcoxon_step_contrasts.csv)

| Metrica | χ² Friedman | df | p-omnibus |
|---------|------------|----|-----------|
| Answer F1 | 44.0 | 3 | 1.5 × 10⁻⁹ |
| OFS | 183.1 | 2 | 1.8 × 10⁻⁴⁰ |
| BERTScore baseline | 483.8 | 2 | 8.6 × 10⁻¹⁰⁶ |
| BERTScore consecutive | 563.7 | 2 | 3.9 × 10⁻¹²³ |
| BLEURT baseline | 341.6 | 2 | 6.6 × 10⁻⁷⁵ |

Tutti gli omnibus sono dominati dal trend monotonico (positivo per BERT consecutive, negativo per gli altri). Contrasti paired step-by-step (n = 297 qid):

| Metrica | Contrasto | Δ medio | rank-biserial | p_holm |
|---------|-----------|---------|---------------|--------|
| **F1** | step0→1 | **−0.147** | **−0.43** | **< 10⁻⁷** |
| F1 | step1→2 | −0.025 | −0.45 | < 10⁻⁷ |
| F1 | step2→3 | −0.014 | −0.29 | 7 × 10⁻⁴ |
| **OFS** | step1→2 | **−0.015** | **−0.62** | **< 10⁻²⁰** |
| OFS | step2→3 | −0.011 | −0.56 | < 10⁻¹⁶ |
| BERTScore baseline | step1→2 | −0.008 | −0.99 | < 10⁻⁴⁹ |
| BERTScore baseline | step2→3 | −0.004 | −0.96 | < 10⁻⁴⁶ |
| BERTScore consecutive | step1→2 | **+0.085** | **+1.00** | < 10⁻⁵⁰ |
| BERTScore consecutive | step2→3 | +0.015 | +0.97 | < 10⁻⁴⁷ |
| BLEURT baseline | step1→2 | −0.027 | −0.92 | < 10⁻⁴² |
| BLEURT baseline | step2→3 | −0.013 | −0.76 | < 10⁻²⁹ |

> **Correzione importante rispetto all'analisi precedente.** Il vecchio `ANALISI_300q.md` riportava step1→2 e step2→3 di F1 come "non significativi" (p_holm = 0.93). Con l'aggregazione per qid usata qui (1 valore medio per qid e step, n = 297) il calo aggiuntivo a step 2 e 3 è statisticamente significativo. La differenza non era nel test ma nella scelta dell'unità: il vecchio test aveva n diversi e likely calcolava ranks su un design non bilanciato. Il rank-biserial conferma che l'effetto è reale ma medio-piccolo.

### S2 — Mixed models e ICC
[stats_v2/s2_mixed_models_coefficients.csv](stats_v2/s2_mixed_models_coefficients.csv) · [stats_v2/s2_icc_qid.csv](stats_v2/s2_icc_qid.csv)

| Modello | ICC(qid) | Interpretazione |
|---------|---------|-----------------|
| F1 (LPM) | 0.402 | 40% della varianza F1 è "quale qid è" |
| OFS | 0.321 | 32% |
| BERTScore baseline | 0.411 | 41% |
| BERTScore consecutive | 0.001 | ≈ 0 → la variabilità è quasi tutta tra catene, non tra qid |
| BLEURT baseline | 0.350 | 35% |

L'ICC≈0 di BERTScore consecutive ha un'interpretazione precisa: la convergenza all'attrattore è un fenomeno **del processo di rewriting**, non di "quale domanda è". Tutte le qid si comportano allo stesso modo nel consecutive drift.

I coefficienti dei modelli misti (vs reference step) confermano i Wilcoxon e aggiungono i contrasti per `n_hop` e `instruction_type`. Per F1, il coefficiente 4-hop vs 2-hop è −0.077 (cioè le 4-hop sono in media 7.7 pp più basse, controllando step e istruzione).

### S3 — Cluster bootstrap CIs (B = 10 000)
[stats_v2/s3_bootstrap_cis.csv](stats_v2/s3_bootstrap_cis.csv)

| Statistica | Stima | 95% CI |
|------------|-------|--------|
| mean F1 @step 0 | 0.362 | [0.314, 0.411] |
| mean F1 @step 1 | 0.215 | [0.187, 0.244] |
| mean F1 @step 3 | 0.177 | [0.152, 0.204] |
| ΔF1 step 0→1 | **−0.147** | **[−0.190, −0.103]** |
| ΔF1 step 0→3 | −0.185 | [−0.230, −0.142] |
| ΔOFS step 1→3 | **−0.025** | **[−0.029, −0.022]** |
| Recovery rate F1≥0.9 | 11.2% | [8.2%, 14.5%] |
| Recovery rate F1≥0.5 | 14.2% | [10.8%, 17.9%] |
| Recovery rate qualsiasi overlap | 21.9% | [17.6%, 26.3%] |
| ΔBERT consecutive step 1→2 | +0.085 | [+0.080, +0.089] |

I CI sono ottenuti ricampionando con replacement 297 qid alla volta (cluster bootstrap), preservando la struttura del disegno paired. I CI per ΔF1 step 0→1 escludono ampiamente zero; quelli per il recovery rate escludono valori intorno al 5% e al 20%, definendo il "vero" recovery in un intervallo ~3 pp.

### S4 — Mediazione causale: step → log(n_tokens) → outcome
[stats_v2/s4_mediation.csv](stats_v2/s4_mediation.csv)

| Outcome | c (totale) | c' (diretto) | a (step→tok) | b (tok→out) | ab (indiretto) | prop mediato (95% CI) |
|---------|-----------|--------------|--------------|-------------|----------------|------------------------|
| **F1** | −0.0580 | −0.0300 | −0.585 | +0.0479 | **−0.0280 [−0.039, −0.017]** | **0.48 [0.31, 0.66]** |
| **OFS** | −0.0127 | −0.0054 | −0.129 | +0.0561 | **−0.0072 [−0.009, −0.006]** | **0.57 [0.43, 0.73]** |

**Lettura.** Un incremento di 1 step abbassa F1 di 5.8 pp in totale (`c`). Di questi:
- ~3.0 pp restano come effetto diretto (cambio di istruzione/style, non riducibile a lunghezza)
- ~2.8 pp sono **mediati dalla compressione**: il modello accorcia il testo (a = −0.59 log-token per step ≈ riduzione del 45% in scala lineare), e per ogni unità di log-token persa F1 cala di 0.048 pp (b).
- **Proporzione mediata = 48% [31%, 66%]** — quasi metà del crollo di F1 è "lunghezza", il resto è qualità intrinseca della riscrittura.

Per OFS la proporzione mediata è ancora più alta: **57% [43%, 73%]**. La fattualità erode in larga parte perché il testo è più corto, non perché diventi falso.

### S5 — Survival analysis del recovery
[stats_v2/s5_km_table.csv](stats_v2/s5_km_table.csv) · [stats_v2/s5_survival_summary.csv](stats_v2/s5_survival_summary.csv) · [stats_v2/s5_logrank_instruction.csv](stats_v2/s5_logrank_instruction.csv)

Su 1 800 catene zero-start (F1 = 0 al testo originale), 202 hanno raggiunto F1 ≥ 0.9 in almeno uno step. Definiamo:
- `duration` = numero di step consecutivi da quel primo recupero in cui F1 resta ≥ 0.9
- `event = 1` se la risposta corretta viene persa entro step 3, `0` se censored a step 3

| Stat | Valore |
|------|--------|
| n episodi | 202 |
| Mediana duration | 2 step |
| % persi entro 1 step | **40.1%** |
| % ancora vivi a step 3 (censored) | 38.6% |

**Tabella Kaplan–Meier (sopravvivenza marginale):**

| Strato | t = 1 | t = 2 | t = 3 |
|--------|-------|-------|-------|
| ALL | 0.713 [0.65, 0.77] | 0.577 [0.50, 0.65] | 0.577 [0.50, 0.65] |
| elaborate | 0.698 | 0.487 | 0.487 |
| formality | 0.733 | 0.570 | 0.570 |
| paraphrase | 0.769 | 0.662 | 0.662 |
| shorten | 0.643 | 0.591 | 0.591 |

**Log-rank multivariato sull'istruzione**: χ² = 2.60, p = 0.46 → le quattro istruzioni non differiscono significativamente nel pattern di sopravvivenza del recovery. La differenza visibile (paraphrase 0.66 vs elaborate 0.49) è entro il rumore di campionamento.

**Conclusione.** Il "recovery" è un fenomeno fragile: dei 202 episodi osservati, più di un terzo viene perso al primo step successivo, e nessuna istruzione protegge meglio. La curva piatta tra t=2 e t=3 (sopravvivenza identica) riflette il fatto che i recuperi che superano lo step 2 tendono a durare fino a fine traccia (non altri eventi).

### S6 — TOST: gli step successivi sono "zero" davvero?
[stats_v2/s6_tost_equivalence.csv](stats_v2/s6_tost_equivalence.csv)

Bound di equivalenza = ±0.02 F1 (≈ 5% del valore @step 0).

| Contrasto | Δ medio | p_TOST | Equivalente a zero? |
|-----------|---------|--------|---------------------|
| step 0 vs 1 | 0.147 | 1.00 | ❌ no |
| step 1 vs 2 | 0.025 | 0.85 | ❌ no |
| step 2 vs 3 | 0.014 | 0.050 | ✅ sì (al pelo) |

**Conclusione.** Il calo step 1→2 (0.025 F1) **eccede** il bound di equivalenza ±0.02: non possiamo concludere che sia "praticamente zero". Solo step 2→3 (0.014) è statisticamente equivalente a zero al 5%. La forma del decadimento è quindi: crollo a step 1, scivolata significativa anche a step 2, plateau da step 3 in poi.

Questo cambia leggermente la narrativa del vecchio doc ("crollo a step 1, poi stabile"): il modello continua a perdere terreno anche al secondo step, in modo più piccolo ma reale.

### S7 — Recovery vs soglia F1
[stats_v2/s7_recovery_sensitivity.csv](stats_v2/s7_recovery_sensitivity.csv)

| Soglia F1 | Recovery rate | 95% CI |
|-----------|--------------|--------|
| > 0 (qualsiasi overlap) | 21.9% | [17.7%, 26.3%] |
| ≥ 0.10 | 20.3% | [16.2%, 24.5%] |
| ≥ 0.25 | 17.6% | [13.9%, 21.5%] |
| ≥ 0.50 | 14.2% | [10.8%, 17.9%] |
| ≥ 0.75 | 11.7% | [8.6%, 15.1%] |
| **≥ 0.90 (≈ EM)** | **11.2%** | **[8.2%, 14.6%]** |
| = 1.0 (EM esatto) | 11.2% | [8.2%, 14.5%] |

**Conclusione.** Il numero "11.2%" che riportiamo come tasso di recovery è robusto da F1 ≥ 0.75 in su: il plateau a 11.2% è tipico di un dataset bimodale (sostanzialmente, recuperare significa F1 = 1). A soglie più lasse (F1 > 0) il numero raddoppia, ma quei "recuperi" sono in larga parte overlap superficiale (es. gold "Charleston" / pred "Richland County"). La conclusione che il recovery genuino è ~11% non dipende dalla scelta esatta della soglia purché stia sopra 0.5.

### S8 — IRR scaffold (manuale, da completare)
[stats_v2/s8_irr_candidates.csv](stats_v2/s8_irr_candidates.csv)

120 osservazioni con F1 = 0 e BLEURT_answer ≥ 0.3 (candidate falsi negativi). Il CSV ha colonna `human_label` vuota: annotando manualmente (0/1) si può calcolare Cohen's κ tra il giudizio umano e l'auto-label F1 = 0. Funzione di calcolo κ pronta nello script, attivabile aggiungendo `--kappa` quando le etichette saranno disponibili. Questa parte è uno **stub onesto**: senza annotazione manuale non esiste IRR.

---

## Cosa è cambiato rispetto all'analisi precedente

1. **Step 1→2 e step 2→3 di F1 sono significativi**, non n.s.. Il vecchio doc usava un'unità diversa (chains o ranks su pool non bilanciato); la nuova analisi paired su per-qid means dà p < 10⁻⁷ con rank-biserial −0.45 e −0.29.
2. **Equivalenza TOST**: solo step 2→3 è equivalente a zero entro ±0.02; step 1→2 non lo è. La conclusione "il danno è tutto al primo step" va attenuata in "il danno è concentrato al primo step ma continua significativamente anche al secondo".
3. **Mediazione causale formale** invece di "togliamo n_tokens e vediamo cosa cambia": 48% del crollo F1 è dovuto alla compressione (CI 31–66%); 57% per OFS (CI 43–73%). Questo trasforma la conclusione "la lunghezza è il confondente principale" in una quantità misurata.
4. **Survival analysis**: il "recovery transiente" del vecchio doc (37.6% perso entro 1 step) era stimato da una singola percentuale; ora abbiamo curva KM con CI, una mediana (2 step) e un log-rank tra istruzioni (p = 0.46, nessuna differenza).
5. **CI cluster-bootstrap** ovunque: ogni Δ chiave ha un intervallo, non solo un p-value.
6. **Sensitivity sulla soglia recovery**: 11.2% è robusto da F1 ≥ 0.75.
7. **ICC(qid) ≈ 0 su BERTScore consecutive** documentata: la convergenza all'attrattore è uniforme tra qid.

---

## Limiti

| Limite | Impatto |
|--------|---------|
| GLMM logit instabile su 14k righe → usiamo LPM | L'inferenza Wilcoxon (non parametrica) è il test primario; LPM è solo riportato per i coefficienti |
| F1 ≥ 0.9 ≈ EM per la struttura bimodale | Da S7 sappiamo che la conclusione regge dal 0.75 in su |
| Mediazione causale assume linearità | Path c, a, c' sono OLS; il check con F1 binario (LPM) dà numeri equivalenti |
| 657 candidati BLEURT≥0.3 / F1=0 non annotati | IRR umano da fare; lo stub è pronto in S8 |
| Solo 3 step | KM censured a step 3: non sappiamo come decade il recovery dopo |

---

## File di output

Tutti i CSV sono in [stats_v2/](stats_v2/). Manifest in [stats_v2/MANIFEST.json](stats_v2/MANIFEST.json).

| File | Sezione | Cosa contiene |
|------|---------|---------------|
| `s0_descriptive_by_step.csv` | S0 | Medie, std, mediana, count per metrica × step |
| `s0_descriptive_by_instruction_step.csv` | S0 | Stesso, per instruction × step |
| `s0_descriptive_by_hop_step.csv` | S0 | Stesso, per hop × step |
| `s1_friedman_omnibus.csv` | S1 | χ², df, p-value per ogni metrica |
| `s1_wilcoxon_step_contrasts.csv` | S1 | Δ paired, rank-biserial, p_raw, p_holm |
| `s2_mixed_models_coefficients.csv` | S2 | Coefficienti, SE, p per ciascun modello misto |
| `s2_icc_qid.csv` | S2 | ICC(qid) per ciascuna metrica |
| `s3_bootstrap_cis.csv` | S3 | Stima + CI 95% per medie, deltas, recovery rate |
| `s4_mediation.csv` | S4 | c, c', a, b, ab, prop_mediated con CI |
| `s5_recovery_episodes.csv` | S5 | Una riga per episodio (qid, instr, run, duration, event) |
| `s5_km_table.csv` | S5 | Sopravvivenza KM per ALL e per ogni istruzione |
| `s5_logrank_instruction.csv` | S5 | Log-rank multivariato fra istruzioni |
| `s5_survival_summary.csv` | S5 | Mediana, % event, % censored |
| `s6_tost_equivalence.csv` | S6 | TOST per ogni contrasto di step su F1 |
| `s7_recovery_sensitivity.csv` | S7 | Recovery rate × 7 soglie con CI |
| `s8_irr_candidates.csv` | S8 | 120 candidati per annotazione manuale (vuoto) |
| `s8_irr_status.csv` | S8 | Stato attuale (AWAITING_LABELS) + istruzioni |
| `s9_rq_summary.csv` | S9 | Riassunto RQ → headline → stima |
