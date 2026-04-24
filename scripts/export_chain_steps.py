import pandas as pd
from pathlib import Path

csv_path = Path("/Users/annasacchet/Desktop/RISULTATI TEST/rewriting_chains32b_factscore.csv")
output_path = csv_path.parent / (csv_path.stem + "_steps.csv")

# Leggi il CSV
df = pd.read_csv(csv_path)

# Mantieni solo le colonne rilevanti per gli step
steps_df = df[["qid", "group", "instruction_type", "run", "step", 
               "n_facts", "n_supported", "n_contradicted", "factscore"]].copy()

# Salva il CSV ordinato per chain
steps_df = steps_df.sort_values(["qid", "group", "instruction_type", "run", "step"])
steps_df.to_csv(output_path, index=False)

print(f"✓ Salvato: {output_path}")
print(f"\nEsempio (prime righe):")
print(steps_df.head(10))
