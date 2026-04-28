"""
Visualizzazione combinata di FactScore e BERTScore.

Crea un PDF multi-pagina mostrando:
1. Scatter plot FactScore vs BERTScore
2. Evolution di entrambe le metriche per step
3. Boxplot per instruction type
4. Stacked bar chart (fatti supportati/no)
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import seaborn as sns
import numpy as np

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "results" / "rewriting_chains32b_factscore_bertscore.csv"
OUTPUT_PDF = REPO_ROOT / "results" / "analisi_factscore_bertscore.pdf"

def prepare_data():
    """Carica e prepara i dati"""
    df = pd.read_csv(CSV_PATH)

    # Legacy CSVs (pre-Consecutive mode) used unsuffixed column names.
    # Map them onto the current Baseline-mode names so older data still loads.
    df = df.rename(columns={
        "bert_precision": "bert_precision_baseline",
        "bert_recall": "bert_recall_baseline",
        "bert_f1": "bert_f1_baseline",
    })

    # Calcola percentuale fatti supportati
    df['pct_supported'] = (df['n_supported'] / df['n_facts'] * 100).round(1)
    df['pct_contradicted'] = (df['n_contradicted'] / df['n_facts'] * 100).round(1)

    return df

def plot_scatter_correlation(df, ax):
    """Plot 1: FactScore vs BERTScore scatter"""
    colors = {'content': '#FF6B6B', 'style': '#4ECDC4'}
    markers = {'elaborate': 'o', 'shorten': 's', 'formality': '^', 'paraphrase': 'D'}
    
    for group in df['group'].unique():
        group_df = df[df['group'] == group]
        for instr_type in group_df['instruction_type'].unique():
            subset = group_df[group_df['instruction_type'] == instr_type]
            
            ax.scatter(
                subset['factscore'],
                subset['bert_f1_baseline'],
                s=150,
                alpha=0.7,
                color=colors[group],
                marker=markers[instr_type],
                label=f"{group}/{instr_type}",
                edgecolors='black',
                linewidth=1
            )
    
    # Diagonal reference line (perfect correlation)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=2, label='Perfect correlation')
    
    # Pearson correlation
    corr = df[['factscore', 'bert_f1_baseline']].corr().iloc[0, 1]
    
    ax.set_xlabel('FactScore (Correttezza Fattuale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BERTScore F1 (Similarità Semantica)', fontsize=12, fontweight='bold')
    ax.set_title(f'FactScore vs BERTScore\nPearson r = {corr:.4f}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    return corr

def plot_evolution_by_instruction(df, ax):
    """Plot 2: Evolution di metriche per instruction type"""
    for instr_type in sorted(df['instruction_type'].unique()):
        subset = df[df['instruction_type'] == instr_type].sort_values('step')
        
        ax.plot(
            subset['step'],
            subset['factscore'],
            marker='o',
            linewidth=2.5,
            markersize=8,
            label=f"{instr_type} (FactScore)",
            alpha=0.8
        )
        ax.plot(
            subset['step'],
            subset['bert_f1_baseline'],
            marker='s',
            linewidth=2.5,
            markersize=8,
            label=f"{instr_type} (BERTScore)",
            alpha=0.6,
            linestyle='--'
        )
    
    ax.set_xlabel('Step nella catena di riscrittura', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Evolution di FactScore vs BERTScore per istruzione', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.0])

def plot_boxplot_by_type(df, ax):
    """Plot 3: Distribuzione metriche per instruction type"""
    # Prepara dati per boxplot
    data_to_plot = []
    labels = []
    positions = []
    colors_list = []
    
    pos = 1
    color_map = {'FactScore': '#1f77b4', 'BERTScore': '#ff7f0e'}
    
    for instr_type in sorted(df['instruction_type'].unique()):
        subset = df[df['instruction_type'] == instr_type]
        
        # FactScore
        data_to_plot.append(subset['factscore'].values)
        labels.append(f"{instr_type}\n(FS)")
        positions.append(pos)
        colors_list.append(color_map['FactScore'])
        pos += 1
        
        # BERTScore
        data_to_plot.append(subset['bert_f1_baseline'].values)
        labels.append(f"{instr_type}\n(BS)")
        positions.append(pos)
        colors_list.append(color_map['BERTScore'])
        pos += 1.5
    
    bp = ax.boxplot(data_to_plot, positions=positions, labels=labels, patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Distribuzione FactScore (blu) vs BERTScore (arancio) per tipo', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.7, 1.0])

def plot_fact_composition(df, ax):
    """Plot 4: Composizione fatti (supportati/contradetti) stacked bar"""
    instr_types = sorted(df['instruction_type'].unique())
    
    n_supported_avg = [df[df['instruction_type'] == t]['pct_supported'].mean() for t in instr_types]
    n_contradicted_avg = [df[df['instruction_type'] == t]['pct_contradicted'].mean() for t in instr_types]
    n_not_supp_avg = [100 - s - c for s, c in zip(n_supported_avg, n_contradicted_avg)]
    
    x = np.arange(len(instr_types))
    width = 0.6
    
    p1 = ax.bar(x, n_supported_avg, width, label='Supportati ✓', color='#2ecc71', alpha=0.8)
    p2 = ax.bar(x, n_not_supp_avg, width, bottom=n_supported_avg, label='Non supportati', color='#95a5a6', alpha=0.8)
    p3 = ax.bar(x, n_contradicted_avg, width, bottom=[s+n for s, n in zip(n_supported_avg, n_not_supp_avg)], 
                label='Contraddetti ❌', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Percentuale fatti (%)', fontsize=11, fontweight='bold')
    ax.set_title('Composizione media fatti per tipo di istruzione', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(instr_types, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Aggiungi percentuali
    for i, (s, n, c) in enumerate(zip(n_supported_avg, n_not_supp_avg, n_contradicted_avg)):
        ax.text(i, s/2, f'{s:.0f}%', ha='center', va='center', fontweight='bold', fontsize=9)
        ax.text(i, s + n/2, f'{n:.0f}%', ha='center', va='center', fontsize=8)
        if c > 0:
            ax.text(i, s + n + c/2, f'{c:.0f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

def create_summary_table(df, fig):
    """Aggiungi tabella sommario"""
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Prepara tabella per instruction type
    summary = []
    for instr_type in sorted(df['instruction_type'].unique()):
        subset = df[df['instruction_type'] == instr_type]
        summary.append([
            instr_type,
            f"{subset['factscore'].mean():.4f}",
            f"{subset['factscore'].std():.4f}",
            f"{subset['bert_f1_baseline'].mean():.4f}",
            f"{subset['bert_f1_baseline'].std():.4f}",
            f"{subset['pct_supported'].mean():.1f}%",
            f"{subset['pct_contradicted'].mean():.1f}%",
        ])
    
    table = ax.table(
        cellText=summary,
        colLabels=['Istruzione', 'FactScore μ', 'FS σ', 'BERTScore μ', 'BS σ', 'Supportati', 'Contraddetti'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Colora header
    for i in range(7):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colora righe alternate
    for i in range(1, len(summary) + 1):
        for j in range(7):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

def main():
    print("="*70)
    print("VISUALIZZAZIONE FACTSCORE + BERTSCORE")
    print("="*70)
    print()
    
    print("📂 Caricamento dati...")
    df = prepare_data()
    print(f"   ✓ {len(df)} righe, {len(df['instruction_type'].unique())} tipi istruzione")
    print()
    
    print("📊 Creazione grafici...")
    
    with PdfPages(str(OUTPUT_PDF)) as pdf:
        # Pagina 1: Scatter + Evolution
        fig = plt.figure(figsize=(14, 10))
        
        ax1 = fig.add_subplot(2, 2, 1)
        corr = plot_scatter_correlation(df, ax1)
        
        ax2 = fig.add_subplot(2, 2, 2)
        plot_evolution_by_instruction(df, ax2)
        
        ax3 = fig.add_subplot(2, 2, 3)
        plot_boxplot_by_type(df, ax3)
        
        ax4 = fig.add_subplot(2, 2, 4)
        plot_fact_composition(df, ax4)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Pagina 2: Tabella sommario
        fig = plt.figure(figsize=(12, 8))
        create_summary_table(df, fig)
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
    
    print(f"✅ PDF salvato: {OUTPUT_PDF}")
    print()
    
    # Stampa sommario
    print("="*70)
    print("SOMMARIO PER TIPO ISTRUZIONE")
    print("="*70)
    print()
    
    for instr_type in sorted(df['instruction_type'].unique()):
        subset = df[df['instruction_type'] == instr_type]
        print(f"{instr_type.upper():15s} | n={len(subset)}")
        print(f"  FactScore    : {subset['factscore'].mean():.4f} ± {subset['factscore'].std():.4f}  (range: {subset['factscore'].min():.4f} - {subset['factscore'].max():.4f})")
        print(f"  BERTScore    : {subset['bert_f1_baseline'].mean():.4f} ± {subset['bert_f1_baseline'].std():.4f}  (range: {subset['bert_f1_baseline'].min():.4f} - {subset['bert_f1_baseline'].max():.4f})")
        print(f"  Supportati   : {subset['pct_supported'].mean():.1f}%")
        print(f"  Contraddetti : {subset['pct_contradicted'].mean():.1f}%")
        print()

if __name__ == "__main__":
    main()
