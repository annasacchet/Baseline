#!/usr/bin/env python3.11
"""
Visualizzazione del conteggio dei token per istruzione e iterazione.

Crea grafici mostrando per ogni run di ogni group (content/style):
1. Evoluzione token count per i 3 step
2. Dettagli per instruction type
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Carica i dati
df = pd.read_csv('/Users/annasacchet/Desktop/Baseline/results/rewriting_chains32b_token_counts.csv')

# Colori e marker
colors = {'elaborate': '#e74c3c', 'shorten': '#3498db', 'formality': '#2ecc71', 'paraphrase': '#f39c12'}
markers = {'elaborate': 'o', 'shorten': 's', 'formality': '^', 'paraphrase': 'D'}

# ===== Genera grafici per ogni group =====
for group_name in sorted(df['group'].unique()):
    group_df = df[df['group'] == group_name]
    runs = sorted(group_df['run'].unique())
    n_runs = len(runs)
    
    # Crea figura con un subplot per ogni run
    fig, axes = plt.subplots(2, (n_runs + 1) // 2, figsize=(5 * ((n_runs + 1) // 2), 10))
    axes = axes.flatten() if n_runs > 1 else [axes]
    
    for ax_idx, run in enumerate(runs):
        run_df = group_df[group_df['run'] == run]
        ax = axes[ax_idx]
        
        # Grafico per ogni instruction type in questo run
        for itype in sorted(run_df['instruction_type'].unique()):
            itype_df = run_df[run_df['instruction_type'] == itype].sort_values('step')
            
            ax.plot(itype_df['step'], itype_df['n_tokens'],
                   marker=markers[itype], markersize=8, linewidth=2.5,
                   label=itype.capitalize(), color=colors[itype])
            ax.scatter(itype_df['step'], itype_df['n_tokens'], 
                      s=100, color=colors[itype], zorder=5, edgecolors='black', linewidth=1)
        
        ax.set_xlabel('Step', fontsize=11, fontweight='bold')
        ax.set_ylabel('Token Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{group_name.upper()} - Run {run}', fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1, 2, 3] if 3 in run_df['step'].values else sorted(run_df['step'].unique()))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Mostra i valori sui punti
        for itype in sorted(run_df['instruction_type'].unique()):
            itype_df = run_df[run_df['instruction_type'] == itype].sort_values('step')
            for step, tokens in zip(itype_df['step'], itype_df['n_tokens']):
                ax.text(step, tokens, f'{int(tokens)}', fontsize=8, ha='center', va='bottom')
    
    # Nascondi gli assi vuoti
    for ax in axes[n_runs:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    output_pdf = f'/Users/annasacchet/Desktop/Baseline/results/token_counts_{group_name}.pdf'
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    print(f"✅ Grafico salvato: {output_pdf}")
    plt.close()

# ===== Grafico comparativo: tutti i run per ogni group =====
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for group_idx, group_name in enumerate(sorted(df['group'].unique())):
    group_df = df[df['group'] == group_name]
    ax = axes[group_idx]
    
    # Plot linea per ogni run-instruction_type
    for run in sorted(group_df['run'].unique()):
        run_df = group_df[group_df['run'] == run]
        for itype in sorted(run_df['instruction_type'].unique()):
            itype_df = run_df[run_df['instruction_type'] == itype].sort_values('step')
            
            ax.plot(itype_df['step'], itype_df['n_tokens'],
                   marker=markers[itype], alpha=0.6, linewidth=1.5,
                   label=f"Run {run} - {itype}", color=colors[itype])
    
    ax.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Token Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Token Counts - {group_name.upper()}', fontsize=13, fontweight='bold')
    ax.set_xticks([0, 1, 2, 3])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig('/Users/annasacchet/Desktop/Baseline/results/token_counts_comparison.pdf', dpi=300, bbox_inches='tight')
print("✅ Grafico comparison salvato: /Users/annasacchet/Desktop/Baseline/results/token_counts_comparison.pdf")
plt.close()

# ===== Statistiche per group e run =====
print("\n📊 Statistiche Token Counts per Group e Run:")
print("=" * 80)
for group_name in sorted(df['group'].unique()):
    print(f"\n{group_name.upper()}:")
    group_df = df[df['group'] == group_name]
    for run in sorted(group_df['run'].unique()):
        run_df = group_df[group_df['run'] == run]
        print(f"\n  Run {run}:")
        for itype in sorted(run_df['instruction_type'].unique()):
            subset = run_df[run_df['instruction_type'] == itype]['n_tokens']
            print(f"    {itype.capitalize()}: Media={subset.mean():.0f}, Min={subset.min()}, Max={subset.max()}")
