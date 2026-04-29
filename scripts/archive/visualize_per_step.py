#!/usr/bin/env python3.11
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(REPO_ROOT / "results" / "rewriting_chains32b_factscore_bertscore.csv")

# Legacy CSVs (pre-Consecutive mode) used unsuffixed column names.
# Map them onto the current Baseline-mode names so older data still loads.
df = df.rename(columns={
    "bert_precision": "bert_precision_baseline",
    "bert_recall": "bert_recall_baseline",
    "bert_f1": "bert_f1_baseline",
})

# Crea figura con 2 subplot: uno per FactScore, uno per BERTScore
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Colori e marker per ogni instruction type
colors = {'elaborate': '#e74c3c', 'shorten': '#3498db', 'formality': '#2ecc71', 'paraphrase': '#f39c12'}
markers = {'elaborate': 'o', 'shorten': 's', 'formality': '^', 'paraphrase': 'D'}

# Plot 1: FactScore per step
for itype in sorted(df['instruction_type'].unique()):
    subset = df[df['instruction_type'] == itype].sort_values('step')
    ax1.plot(subset['step'], subset['factscore'], 
             marker=markers[itype], markersize=8, linewidth=2.5,
             label=itype.capitalize(), color=colors[itype])
    ax1.scatter(subset['step'], subset['factscore'], s=100, color=colors[itype], zorder=5)

ax1.set_xlabel('Step (Iterazione)', fontsize=12, fontweight='bold')
ax1.set_ylabel('FactScore', fontsize=12, fontweight='bold')
ax1.set_title('Evoluzione FactScore per iterazione', fontsize=13, fontweight='bold')
ax1.set_xticks([1, 2, 3])
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')
ax1.set_ylim([0.8, 1.0])

# Plot 2: BERTScore per step
for itype in sorted(df['instruction_type'].unique()):
    subset = df[df['instruction_type'] == itype].sort_values('step')
    ax2.plot(subset['step'], subset['bert_f1_baseline'], 
             marker=markers[itype], markersize=8, linewidth=2.5,
             label=itype.capitalize(), color=colors[itype])
    ax2.scatter(subset['step'], subset['bert_f1_baseline'], s=100, color=colors[itype], zorder=5)

ax2.set_xlabel('Step (Iterazione)', fontsize=12, fontweight='bold')
ax2.set_ylabel('BERTScore F1', fontsize=12, fontweight='bold')
ax2.set_title('Evoluzione BERTScore F1 per iterazione', fontsize=13, fontweight='bold')
ax2.set_xticks([1, 2, 3])
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')
ax2.set_ylim([0.75, 1.0])

plt.tight_layout()
out_pdf = REPO_ROOT / "results" / "evoluzione_per_step.pdf"
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
print(f"Grafico salvato: {out_pdf}")

# Mostra anche i valori numerici
print("\n" + "="*70)
print("EVOLUZIONE PER STEP - DETTAGLI NUMERICI")
print("="*70)
for itype in sorted(df['instruction_type'].unique()):
    print(f"\n{itype.upper()}:")
    subset = df[df['instruction_type'] == itype][['step', 'factscore', 'bert_f1_baseline']].sort_values('step')
    for _, row in subset.iterrows():
        print(f"  Step {int(row['step'])}: FactScore={row['factscore']:.4f}, BERTScore={row['bert_f1_baseline']:.4f}")
