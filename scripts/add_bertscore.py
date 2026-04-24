"""
Add BERTScore as complementary metric to FactScore evaluation.

BERTScore misura la similarità semantica token-level tramite BERT embeddings.
A differenza di FactScore (correttezza fattuale), misura quanto il candidato
rimane "semanticamente simile" al riferimento.

Questo script legge i risultati FactScore e aggiunge le metriche BERTScore.
"""

import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm
from bert_score import score as compute_bert_score

# Paths
CSV_PATH = Path("/Users/annasacchet/Desktop/RISULTATI TEST/rewriting_chains32b.csv")
FACTSCORE_PATH = Path("/Users/annasacchet/Desktop/RISULTATI TEST/rewriting_chains32b_factscore.csv")
OUTPUT_PATH = Path("/Users/annasacchet/Desktop/RISULTATI TEST/rewriting_chains32b_factscore_bertscore.csv")

def load_source_texts(csv_path):
    """Carica il CSV originale per avere accesso ai testi step 0 (source)"""
    df = pd.read_csv(csv_path)
    # Raggruppa per chain, prendi step 0
    sources = {}
    for (qid, group, instruction_type, run), group_df in df.groupby(
        ["qid", "group", "instruction_type", "run"]
    ):
        step0 = group_df[group_df["step"] == 0]
        if not step0.empty:
            sources[(qid, group, instruction_type, run)] = step0.iloc[0]["text"]
    return sources

def load_candidate_texts(csv_path):
    """Carica i testi candidati dal CSV originale"""
    df = pd.read_csv(csv_path)
    return df

def compute_bertscore_batch(references, candidates, lang='en', model='roberta-large'):
    """
    Calcola BERTScore per batch di testi.
    
    Args:
        references: lista di stringhe (step 0 source)
        candidates: lista di stringhe (step > 0)
        lang: codice lingua
        model: modello BERT
    
    Returns:
        tuple(P, R, F1) come tensori
    """
    try:
        P, R, F1 = compute_bert_score(
            candidates,
            references,
            lang=lang,
            model_type=model,
            num_layers=17,  # RoBERTa default layer
            batch_size=8,
            device='cpu'  # CPU mode
        )
        return P, R, F1
    except Exception as e:
        print(f"⚠️  Errore BERTScore: {e}")
        return None, None, None

def main():
    print("=" * 70)
    print("AGGIUNTA BERTScore A FactScore")
    print("=" * 70)
    print()
    
    # Carica dati
    print("📂 Caricamento dati...")
    if not FACTSCORE_PATH.exists():
        print(f"❌ File non trovato: {FACTSCORE_PATH}")
        print("   Esegui prima factscore_eval.py")
        return
    
    factscore_df = pd.read_csv(FACTSCORE_PATH)
    all_texts = load_candidate_texts(CSV_PATH)
    sources = load_source_texts(CSV_PATH)
    
    print(f"   ✓ FactScore: {len(factscore_df)} righe")
    print(f"   ✓ Source texts: {len(sources)} chain uniche")
    print(f"   ✓ All texts: {len(all_texts)} righe")
    print()
    
    # Join per recuperare i testi candidati dal CSV originale
    merged = factscore_df.copy()
    merged['text'] = merged.apply(
        lambda row: all_texts[
            (all_texts['qid'] == row['qid']) &
            (all_texts['group'] == row['group']) &
            (all_texts['instruction_type'] == row['instruction_type']) &
            (all_texts['run'] == row['run']) &
            (all_texts['step'] == row['step'])
        ]['text'].iloc[0] if len(all_texts[
            (all_texts['qid'] == row['qid']) &
            (all_texts['group'] == row['group']) &
            (all_texts['instruction_type'] == row['instruction_type']) &
            (all_texts['run'] == row['run']) &
            (all_texts['step'] == row['step'])
        ]) > 0 else None,
        axis=1
    )
    
    # Calcola BERTScore
    print("🔄 Calcolo BERTScore...")
    print(f"   Modello: roberta-large (layer 17)")
    print(f"   Lingua: italiano")
    print()
    
    bert_p, bert_r, bert_f1 = [], [], []
    
    t_start = time.time()
    for i, (idx, row) in enumerate(merged.iterrows(), start=1):
        chain_id = (row["qid"], row["group"], row["instruction_type"], row["run"])
        source = sources.get(chain_id)
        candidate = row["text"]
        
        if pd.isna(source) or pd.isna(candidate):
            bert_p.append(None)
            bert_r.append(None)
            bert_f1.append(None)
            continue
        
        # BERTScore per questa coppia
        P, R, F1 = compute_bert_score(
            [candidate],
            [source],
            lang='en',
            model_type='roberta-large',
            num_layers=17,
            batch_size=1,
            device='cpu'
        )
        
        if P is not None:
            bert_p.append(P.item())
            bert_r.append(R.item())
            bert_f1.append(F1.item())
        else:
            bert_p.append(None)
            bert_r.append(None)
            bert_f1.append(None)
        
        if i % 5 == 0:
            elapsed = time.time() - t_start
            eta = (elapsed / i) * (len(merged) - i)
            print(f"   [{i}/{len(merged)}] {row['group']}/{row['instruction_type']}/step{row['step']} "
                  f"| F1={bert_f1[-1]:.4f} | ETA: {eta:.0f}s")
    
    print()
    print(f"✓ Tempo totale: {(time.time() - t_start):.1f}s")
    print()
    
    # Aggiungi colonne BERTScore
    factscore_df["bert_precision"] = bert_p
    factscore_df["bert_recall"] = bert_r
    factscore_df["bert_f1"] = bert_f1
    
    # Salva
    factscore_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Salvato: {OUTPUT_PATH}")
    print()
    
    # Statistiche
    print("=" * 70)
    print("STATISTICHE BERTScore")
    print("=" * 70)
    print()
    print(factscore_df[["group", "instruction_type", "bert_f1"]].groupby(
        ["group", "instruction_type"]
    )["bert_f1"].agg(["mean", "std", "min", "max"]).round(4))
    print()
    
    # Correlazione FactScore vs BERTScore
    print("=" * 70)
    print("CORRELAZIONE FactScore vs BERTScore")
    print("=" * 70)
    print()
    
    valid_mask = factscore_df["bert_f1"].notna() & factscore_df["factscore"].notna()
    if valid_mask.sum() > 0:
        corr = factscore_df[valid_mask][["factscore", "bert_f1"]].corr()
        print(f"Pearson correlation: {corr.iloc[0, 1]:.4f}")
    else:
        print("Nessun dato valido per calcolare correlazione")
    print()

if __name__ == "__main__":
    main()
