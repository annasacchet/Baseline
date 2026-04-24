# Baseline - Text Rewriting Evaluation Pipeline

Progetto di valutazione multimetrica per analizzare le trasformazioni di testo (riscrittura, accorciamento, parafrasi, formalizzazione).

## Struttura del Progetto

```
Baseline/
├── scripts/          # Script di elaborazione e visualizzazione
│   ├── export_chain_steps.py           # Estrae gli step dalle chain
│   ├── factscore_eval.py               # Valuta FactScore (fedeltà fattuale)
│   ├── add_bertscore.py                # Aggiunge BERTScore (similarità semantica)
│   ├── visualize_factscore_bertscore.py # Visualizzazioni combinate
│   └── visualize_per_step.py           # Grafici per iterazione
│
├── results/          # CSV, PDF e risultati dell'analisi
│   ├── rewriting_chains32b_factscore_bertscore.csv  # Metriche finali
│   ├── analisi_factscore_bertscore.pdf              # Visualizzazione multipla
│   └── evoluzione_per_step.pdf                      # Evoluzione per iterazione
│
└── README.md         # Questo file
```

## Metriche

### FactScore
- **Definizione**: Misura la fedeltà fattuale (quanti fatti del testo originale rimangono supportati)
- **Range**: 0.0 - 1.0 (1.0 = tutti i fatti supportati)
- **Composizione**: Conta fatti supportati, non-supportati e contraddetti

### BERTScore
- **Definizione**: Misura la similarità semantica token-level usando embeddings RoBERTa
- **Range**: 0.0 - 1.0 (1.0 = identico semanticamente)
- **Componenti**: Precision, Recall, F1

## Istruzioni di Trasformazione

1. **elaborate** - Aggiunge dettagli mantenendo fedeltà
   - FactScore: 0.894 ± 0.013
   - BERTScore F1: 0.884 ± 0.016

2. **formality** - Rende il testo più formale
   - FactScore: 0.912 ± 0.014
   - BERTScore F1: 0.915 ± 0.029

3. **paraphrase** - Riscrive mantenendo il significato
   - FactScore: 0.896 ± 0.043
   - BERTScore F1: 0.862 ± 0.043

4. **shorten** - Riduce il contenuto mantenendo il senso
   - FactScore: 0.877 ± 0.032
   - BERTScore F1: 0.789 ± 0.013

## Come Usare

### Esecuzione completa:
```bash
cd Baseline/scripts
export OPENAI_API_KEY="your-key-here"
python3.11 export_chain_steps.py
python3.11 factscore_eval.py
python3.11 add_bertscore.py
python3.11 visualize_factscore_bertscore.py
python3.11 visualize_per_step.py
```

### Generare solo visualizzazioni:
```bash
python3.11 visualize_factscore_bertscore.py
python3.11 visualize_per_step.py
```

## Risultati

- **Correlazione FactScore vs BERTScore**: r = 0.7121 (moderata-forte)
- **Trend**: Tutte le trasformazioni mostrano degradazione al crescere degli step
- **Osservazione**: Le trasformazioni sono più efficaci al primo step

## Data

Run: 1 (run_id = 0)
Righe: 12 (4 istruzioni × 3 step)
Total fatti analizzati: 579-688 per istruzione
