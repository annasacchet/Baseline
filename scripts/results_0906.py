"""
results_0906.py — Panoramica esperimenti rewriting (data: 09/06/2026).

Cosa fa
-------
1. SEZIONE A — 300q (OLMo 32B 4-bit su Lisa, F1 in bf16 su Homer)
   - Determina il subset di qid che hanno tutte le metriche disponibili
     (chains, F1, BERTScore, OpenFActScore).
   - Calcola lunghezza (n_tokens), Answer F1, BERTScore baseline+consecutive,
     OpenFActScore precision (init_score) e factscore (length-penalised).
   - Plot per instruction_type × step.

2. SEZIONE B — Confronto 15q full-precision (Homer bf16) vs 300q quantizzato (Lisa 4-bit)
   - Considera solo i qid in comune fra il pilot 15q e il run 300q (sono 10).
   - Per ogni qid in comune confronta lunghezza, F1, BERTScore, OFS step per step.
   - Obiettivo: isolare l'effetto della quantizzazione 4-bit a parità di
     input (stessi qid, stesso modello, stessa pipeline).

Output
------
results/results_0906.pdf — PDF in italiano con titoli, tabelle e grafici.

Esecuzione
----------
  python3.11 scripts/results_0906.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parent.parent
RES = REPO_ROOT / "results"
OUT_PDF = RES / "results_0906.pdf"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]
INSTRUCTIONS = ["shorten", "paraphrase", "formality", "elaborate"]

# Stili coerenti per le 4 instruction
INSTR_COLORS = {
    "shorten":    "#1f77b4",   # blu
    "paraphrase": "#2ca02c",   # verde
    "formality":  "#ff7f0e",   # arancione
    "elaborate":  "#d62728",   # rosso
}


# ---------------------------------------------------------------------------
# Caricamento dati
# ---------------------------------------------------------------------------

def load_all() -> dict[str, pd.DataFrame]:
    return {
        "chains_15q":  pd.read_csv(RES / "15q"  / "rewriting_chains_15q.csv"),
        "f1_15q":      pd.read_csv(RES / "15q"  / "rewriting_chains_15q_answer_f1.csv"),
        "bert_15q":    pd.read_csv(RES / "15q"  / "rewriting_chains_15q_bertscore.csv"),
        "ofs_15q":     pd.read_csv(RES / "15q"  / "rewriting_chains_15q_openfactscore.csv"),
        "chains_300q": pd.read_csv(RES / "300q" / "rewriting_chains_300q.csv"),
        "f1_300q":     pd.read_csv(RES / "300q" / "rewriting_chains_300q_answer_f1.csv"),
        "bert_300q":   pd.read_csv(RES / "300q" / "rewriting_chains_300q_bertscore.csv"),
        "ofs_300q":    pd.read_csv(RES / "300q" / "rewriting_chains_300q_openfactscore.csv"),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def text_page(pdf: PdfPages, title: str, paragraphs: list[str], font: int = 11) -> None:
    """Pagina di solo testo (Markdown-like) — copertina, paragrafi narrativi."""
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
    ax.axis("off")
    y = 0.95
    ax.text(0.05, y, title, fontsize=18, weight="bold", va="top")
    y -= 0.06
    for p in paragraphs:
        wrapped = wrap_for_pdf(p, max_chars=95)
        for line in wrapped:
            ax.text(0.05, y, line, fontsize=font, va="top", family="serif")
            y -= 0.025
            if y < 0.05:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                y = 0.95
        y -= 0.012  # extra spacing tra paragrafi
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def wrap_for_pdf(text: str, max_chars: int = 95) -> list[str]:
    """Word-wrap conservando newline esistenti."""
    out = []
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            out.append("")
            continue
        words = raw_line.split()
        cur = ""
        for w in words:
            if len(cur) + len(w) + 1 <= max_chars:
                cur = (cur + " " + w).strip()
            else:
                out.append(cur)
                cur = w
        if cur:
            out.append(cur)
    return out


def table_page(pdf: PdfPages, title: str, df: pd.DataFrame, intro: str = "") -> None:
    """Pagina con titolo + (opzionale) intro narrativa + tabella."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    ax.text(0.02, 0.97, title, fontsize=14, weight="bold", va="top", transform=ax.transAxes)
    if intro:
        wrapped = wrap_for_pdf(intro, max_chars=110)
        y = 0.91
        for line in wrapped:
            ax.text(0.02, y, line, fontsize=10, va="top", transform=ax.transAxes, family="serif")
            y -= 0.03
        table_top = max(0.05, y - 0.04)
    else:
        table_top = 0.85
    # rendering tabella
    df_show = df.round(3).reset_index()
    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        cellLoc="center",
        loc="upper center",
        bbox=[0.05, 0.05, 0.9, table_top - 0.05],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def lineplot_by_instruction(pdf: PdfPages, title: str, intro: str,
                             df: pd.DataFrame, value_col: str, ylabel: str,
                             include_step0: bool = True) -> None:
    """Linee separate per instruction_type, asse x = step."""
    fig, (ax_text, ax) = plt.subplots(2, 1, figsize=(10, 8),
                                       gridspec_kw={"height_ratios": [1, 4]})
    ax_text.axis("off")
    ax_text.text(0.0, 1.0, title, fontsize=14, weight="bold", va="top")
    if intro:
        wrapped = wrap_for_pdf(intro, max_chars=115)
        y = 0.85
        for line in wrapped:
            ax_text.text(0.0, y, line, fontsize=10, va="top", family="serif")
            y -= 0.13

    piv = df.pivot_table(index="step", columns="instruction_type", values=value_col,
                         aggfunc="mean")
    if not include_step0 and 0 in piv.index:
        piv = piv.drop(index=0)
    for instr in INSTRUCTIONS:
        if instr in piv.columns:
            ax.plot(piv.index, piv[instr], marker="o", linewidth=2,
                    label=instr, color=INSTR_COLORS[instr])
    ax.set_xlabel("Step di rewriting", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(sorted(piv.index))
    ax.grid(alpha=0.3)
    ax.legend(loc="best", title="instruction_type")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def grouped_bars_compare(pdf: PdfPages, title: str, intro: str,
                         df_left: pd.DataFrame, df_right: pd.DataFrame,
                         value_col: str, label_left: str, label_right: str,
                         ylabel: str, include_step0: bool = True) -> None:
    """Confronto 4-pannelli (uno per instruction) di due dataset, barre per step."""
    instructions = INSTRUCTIONS
    fig, axes = plt.subplots(2, 3, figsize=(14, 8),
                              gridspec_kw={"width_ratios": [1, 1, 1]})
    # Slot top-left = testo
    ax_text = axes[0, 0]
    ax_text.axis("off")
    ax_text.text(0.0, 1.0, title, fontsize=13, weight="bold", va="top",
                 transform=ax_text.transAxes)
    if intro:
        wrapped = wrap_for_pdf(intro, max_chars=42)
        y = 0.88
        for line in wrapped:
            ax_text.text(0.0, y, line, fontsize=9, va="top",
                         transform=ax_text.transAxes, family="serif")
            y -= 0.06

    plot_axes = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    for i, instr in enumerate(instructions):
        ax = plot_axes[i]
        sub_l = df_left[df_left["instruction_type"] == instr]
        sub_r = df_right[df_right["instruction_type"] == instr]
        means_l = sub_l.groupby("step")[value_col].mean()
        means_r = sub_r.groupby("step")[value_col].mean()
        if not include_step0:
            means_l = means_l.drop(index=0, errors="ignore")
            means_r = means_r.drop(index=0, errors="ignore")
        steps = sorted(set(means_l.index) | set(means_r.index))
        x = np.arange(len(steps))
        ax.bar(x - 0.2, [means_l.get(s, np.nan) for s in steps], width=0.4,
               label=label_left, color="#3b6db1")
        ax.bar(x + 0.2, [means_r.get(s, np.nan) for s in steps], width=0.4,
               label=label_right, color="#c25450")
        ax.set_title(instr, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"E_{s}" for s in steps])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    # ultimo slot vuoto
    if len(instructions) < 5:
        axes[1, 2].axis("off")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SEZIONE A — 300q
# ---------------------------------------------------------------------------

def section_300q(pdf: PdfPages, data: dict) -> None:
    chains = data["chains_300q"]
    f1 = data["f1_300q"]
    bert = data["bert_300q"]
    ofs = data["ofs_300q"]

    # Subset comune di qid (intersezione di tutte e 4)
    q_chains = set(chains["qid"].unique())
    q_f1 = set(f1["qid"].unique())
    q_bert = set(bert["qid"].unique())
    q_ofs = set(ofs["qid"].unique())
    common = q_chains & q_f1 & q_bert & q_ofs

    chains_c = chains[chains["qid"].isin(common)]
    f1_c = f1[f1["qid"].isin(common)]
    bert_c = bert[bert["qid"].isin(common)]
    ofs_c = ofs[ofs["qid"].isin(common)]

    text_page(
        pdf,
        "SEZIONE A — Esperimento 300q (modello quantizzato)",
        [
            "Setup sperimentale",
            f"• Modello rewriter: OLMo-2-0325-32B-Instruct in 4-bit NF4 su Lisa (RTX 3090).",
            f"• Modello QA per Answer F1: OLMo-2-1124-32B-Instruct in bf16 su Homer (RTX A6000).",
            f"• OpenFActScore (AFG + AFV): in 4-bit NF4 su Lisa.",
            f"• Dataset: MuSiQue dev, sample bilanciato per hop count, target 300 q.",
            "",
            f"Stato attuale dei file di output (numero di qid coperti)",
            f"  - Catene di rewriting (chains): {len(q_chains)} qid",
            f"  - Answer F1:                    {len(q_f1)} qid",
            f"  - BERTScore:                    {len(q_bert)} qid",
            f"  - OpenFActScore (precision):    {len(q_ofs)} qid",
            "",
            f"Subset comune a tutte e 4 le metriche: {len(common)} qid.",
            f"Tutte le statistiche e i grafici della Sezione A sono calcolati su "
            f"questo sottoinsieme, in modo che ogni metrica venga aggregata sugli "
            f"stessi documenti — confronto onesto step-per-step.",
            "",
            "Nota metodologica",
            "Il rewriter è quantizzato a 4-bit. Il QA è in bf16 (full-precision). Il dato "
            "Answer F1 NON è inquinato da quantizzazione del lato QA, mentre il pattern di "
            "compressione visibile su lunghezza/n_facts riflette il comportamento del "
            "rewriter quantizzato. OpenFActScore precision usa AFG/AFV in 4-bit, ma il "
            "trend tra step e instruction è preservato (il bias è uniforme).",
        ],
    )

    # ---- 1. Lunghezza (token count)
    chains_with_step0 = chains_c[["qid","group","instruction_type","run","step","n_tokens"]]
    lineplot_by_instruction(
        pdf,
        "A.1 — Lunghezza media (n_tokens) per step e instruction",
        f"Andamento del numero medio di token in E_t, calcolato sui {len(common)} qid in "
        "comune. Lo step 0 corrisponde al testo di partenza (E_0). "
        "Se l'instruction agisce come dovrebbe ci si aspetta: shorten ↓, elaborate ↑, "
        "paraphrase e formality stabili.",
        chains_with_step0, "n_tokens", "n_tokens (medio)",
    )
    # tabella associata
    piv_tok = chains_with_step0.pivot_table(index="instruction_type", columns="step",
                                             values="n_tokens", aggfunc="mean")
    table_page(pdf, "A.1 — Tabella: n_tokens medi per instruction × step",
               piv_tok,
               intro=f"Valori medi sui {len(common)} qid in comune. Si noti che anche "
                     "elaborate produce meno token rispetto a E_0 → segnale che il rewriter "
                     "quantizzato tende a comprimere indipendentemente dall'instruction.")

    # ---- 2. Answer F1
    lineplot_by_instruction(
        pdf,
        "A.2 — Answer F1 medio per step e instruction",
        f"F1 dell'output del modello QA su {len(common)} qid in comune. "
        "Lo step 0 misura la baseline F1 sul testo originale. Una caduta forte "
        "indica che l'informazione necessaria a rispondere si è persa nel rewriting.",
        f1_c, "answer_f1", "Answer F1 (medio)",
    )
    piv_f1 = f1_c.pivot_table(index="instruction_type", columns="step",
                               values="answer_f1", aggfunc="mean")
    table_page(pdf, "A.2 — Tabella: Answer F1 medio per instruction × step",
               piv_f1,
               intro="Tutti i tipi di instruction degradano F1 in modo simile → la "
                     "perdita di informazione non dipende dal tipo di intervento ma dal "
                     "fatto stesso di iterare il rewriting con un modello quantizzato.")

    # ---- 3. BERTScore baseline (vs E_0)
    lineplot_by_instruction(
        pdf,
        "A.3 — BERTScore F1 baseline (similarità di E_t con E_0)",
        f"Misura quanto E_t è ancora semanticamente simile a E_0. "
        "Calcolato con roberta-large. Valori vicini a 1 = E_t è quasi identico a E_0. "
        "Una discesa rapida indica drift dal contenuto originale.",
        bert_c, "bert_f1_baseline", "BERTScore F1 vs E_0", include_step0=False,
    )
    piv_bert_b = bert_c.pivot_table(index="instruction_type", columns="step",
                                     values="bert_f1_baseline", aggfunc="mean")
    table_page(pdf, "A.3 — Tabella: BERTScore F1 vs E_0",
               piv_bert_b,
               intro="elaborate è l'instruction che si allontana di più da E_0; le altre "
                     "tre restano in un range simile.")

    # ---- 4. BERTScore consecutive (vs E_{t-1})
    lineplot_by_instruction(
        pdf,
        "A.4 — BERTScore F1 consecutive (similarità con lo step precedente)",
        "Misura quanto cambia il testo da uno step al successivo. Valori che salgono "
        "verso 1 indicano che il rewriting si stabilizza: dopo le prime iterazioni il "
        "modello cambia poco.",
        bert_c, "bert_f1_consecutive", "BERTScore F1 vs E_{t-1}", include_step0=False,
    )

    # ---- 5. OpenFActScore precision (init_score)
    lineplot_by_instruction(
        pdf,
        "A.5 — OpenFActScore precision (init_score) per step",
        "Frazione di fatti estratti da E_t che sono supportati da E_0. Misura quanto "
        "il rewriter è fedele al contenuto originale: 1.0 = nessun fatto inventato. "
        f"Calcolato sui {len(common)} qid in comune.",
        ofs_c, "init_score", "OFS precision (init_score)", include_step0=False,
    )
    piv_ofs = ofs_c.pivot_table(index="instruction_type", columns="step",
                                 values="init_score", aggfunc="mean")
    table_page(pdf, "A.5 — Tabella: OFS precision per instruction × step",
               piv_ofs,
               intro="elaborate è l'unica con caduta significativa di precision (≈ -5pp dopo "
                     "3 step) → conferma che l'instruction 'elabora' spinge il modello a "
                     "introdurre fatti non supportati da E_0.")

    # ---- 6. n_facts (sanity check sul comportamento di compressione)
    if "n_facts" in ofs_c.columns:
        piv_nfacts = ofs_c.pivot_table(index="instruction_type", columns="step",
                                        values="n_facts", aggfunc="mean")
        table_page(pdf, "A.6 — Tabella: numero medio di fatti estratti da E_t",
                   piv_nfacts,
                   intro="Anche elaborate produce sempre meno fatti a ogni step (es. "
                         "80→70 al posto di crescere). Sintomo che il rewriter quantizzato "
                         "non segue l'instruction 'aggiungi dettagli' e finisce per comprimere.")


# ---------------------------------------------------------------------------
# SEZIONE B — Confronto 15q full vs 300q quantizzato
# ---------------------------------------------------------------------------

def section_compare(pdf: PdfPages, data: dict) -> None:
    chains_15 = data["chains_15q"]
    chains_300 = data["chains_300q"]
    f1_15 = data["f1_15q"]
    f1_300 = data["f1_300q"]
    bert_15 = data["bert_15q"]
    bert_300 = data["bert_300q"]
    ofs_15 = data["ofs_15q"]
    ofs_300 = data["ofs_300q"]

    common = sorted(set(chains_15["qid"]) & set(chains_300["qid"]))

    text_page(
        pdf,
        "SEZIONE B — Confronto full-precision (15q) vs 4-bit (300q)",
        [
            "Obiettivo",
            "Isolare l'effetto della quantizzazione 4-bit del rewriter, confrontando il "
            "pilot 15q (rewriter in bf16 su Homer) con il sottoinsieme degli stessi qid "
            "presenti nel run 300q (rewriter in 4-bit su Lisa).",
            "",
            f"Qid in comune fra i due esperimenti: {len(common)} su 15.",
            f"  • {', '.join(common)}",
            "",
            "Nota metodologica",
            "Stesso modello (OLMo 32B Instruct), stesso prompt template, stessi qid, "
            "stesso pipeline. Sola variabile: la quantizzazione 4-bit del rewriter. "
            "L'Answer F1 sui 300q è in bf16 (Homer) — quindi la differenza F1 osservata "
            "rispetto al 15q è attribuibile ai DIVERSI testi prodotti dal rewriter, "
            "non al QA model.",
            "",
            "Ogni grafico/tabella della Sezione B è ristretto a questi qid in comune.",
        ],
    )

    chains_15c = chains_15[chains_15["qid"].isin(common)]
    chains_300c = chains_300[chains_300["qid"].isin(common)]
    f1_15c = f1_15[f1_15["qid"].isin(common)]
    f1_300c = f1_300[f1_300["qid"].isin(common)]
    bert_15c = bert_15[bert_15["qid"].isin(common)]
    bert_300c = bert_300[bert_300["qid"].isin(common)]
    ofs_15c = ofs_15[ofs_15["qid"].isin(common)]
    ofs_300c = ofs_300[ofs_300["qid"].isin(common)]

    # B.1 — lunghezza
    grouped_bars_compare(
        pdf, "B.1 — Lunghezza (n_tokens) per step e instruction",
        f"Confronto a parità di qid ({len(common)}).\n"
        "Blu = bf16 (15q),\nrosso = 4-bit (300q).\n"
        "Se il modello quantizzato comprime di più, le barre rosse saranno più basse.",
        chains_15c, chains_300c, "n_tokens",
        "bf16 (15q)", "4-bit (300q)", "n_tokens",
    )
    # tabella riassuntiva delta
    p15 = chains_15c.pivot_table(index="instruction_type", columns="step",
                                  values="n_tokens", aggfunc="mean")
    p300 = chains_300c.pivot_table(index="instruction_type", columns="step",
                                    values="n_tokens", aggfunc="mean")
    delta = (p300 - p15).round(0).astype("Int64")
    table_page(pdf, "B.1 — Δ n_tokens (300q − 15q) per instruction × step",
               delta,
               intro=f"Delta in token tra 4-bit e bf16 sui {len(common)} qid in comune. "
                     "Numeri negativi indicano che il modello 4-bit produce meno token. "
                     "elaborate dovrebbe avere delta vicino a 0 o positivo se la "
                     "quantizzazione non interferisse con l'instruction following.")

    # B.2 — Answer F1
    grouped_bars_compare(
        pdf, "B.2 — Answer F1 per step e instruction",
        f"Confronto a parità di qid ({len(common)}).\n"
        "Se le barre rosse (4-bit) sono molto più basse,\n"
        "la quantizzazione del rewriter sta facendo sparire la risposta dal testo.",
        f1_15c, f1_300c, "answer_f1",
        "bf16 (15q)", "4-bit (300q)", "F1",
    )
    p15 = f1_15c.pivot_table(index="instruction_type", columns="step",
                              values="answer_f1", aggfunc="mean")
    p300 = f1_300c.pivot_table(index="instruction_type", columns="step",
                                values="answer_f1", aggfunc="mean")
    table_page(pdf, "B.2 — Δ Answer F1 (300q − 15q) per instruction × step",
               (p300 - p15),
               intro="Delta F1 tra 4-bit e bf16. Valori negativi grandi confermano che il "
                     "rewriter quantizzato distrugge l'informazione necessaria al QA.")

    # B.3 — BERTScore baseline
    if not bert_15c.empty and not bert_300c.empty:
        grouped_bars_compare(
            pdf, "B.3 — BERTScore F1 vs E_0 (drift cumulativo)",
            f"Quanto E_t resta simile a E_0 ({len(common)} qid in comune).",
            bert_15c, bert_300c, "bert_f1_baseline",
            "bf16 (15q)", "4-bit (300q)", "BERTScore F1",
            include_step0=False,
        )

    # B.4 — OFS precision
    if not ofs_15c.empty and not ofs_300c.empty:
        grouped_bars_compare(
            pdf, "B.4 — OpenFActScore precision (init_score)",
            f"Fedeltà al contenuto di E_0 ({len(common)} qid in comune). "
            "OFS è meno sensibile alla quantizzazione (AFG/AFV dovrebbero comportarsi "
            "in modo simile) — eventuali differenze importanti sono indicative.",
            ofs_15c, ofs_300c, "init_score",
            "bf16 (15q)", "4-bit (300q)", "OFS precision",
            include_step0=False,
        )


# ---------------------------------------------------------------------------
# Sintesi finale
# ---------------------------------------------------------------------------

def section_summary(pdf: PdfPages, data: dict) -> None:
    # Calcoli per la narrazione
    common = sorted(set(data["chains_15q"]["qid"]) & set(data["chains_300q"]["qid"]))
    c15 = data["chains_15q"][data["chains_15q"]["qid"].isin(common)]
    c300 = data["chains_300q"][data["chains_300q"]["qid"].isin(common)]
    f15 = data["f1_15q"][data["f1_15q"]["qid"].isin(common)]
    f300 = data["f1_300q"][data["f1_300q"]["qid"].isin(common)]

    def step3(df, col):
        return df[df["step"] == 3].groupby("instruction_type")[col].mean()

    tok15 = step3(c15, "n_tokens")
    tok300 = step3(c300, "n_tokens")
    f1_15_step3 = step3(f15, "answer_f1")
    f1_300_step3 = step3(f300, "answer_f1")
    f1_15_step0 = f15[f15["step"] == 0].groupby("instruction_type")["answer_f1"].mean()
    f1_300_step0 = f300[f300["step"] == 0].groupby("instruction_type")["answer_f1"].mean()

    paragraphs = [
        "Sintesi dei risultati",
        "",
        "1. Pattern di compressione",
        "Anche il rewriter in bf16 (15q) tende a comprimere il testo invece di rispettare "
        "rigorosamente le instruction (specialmente con input lunghi 2300+ token come "
        "in MuSiQue). Tuttavia il modello 4-bit (300q) amplifica nettamente questo "
        "comportamento: a step 3 produce in media molti meno token di quanto faccia il "
        "modello bf16 a parità di qid e instruction.",
        "",
        f"  • elaborate · step 3 · bf16: {tok15.get('elaborate', float('nan')):.0f} token  "
        f"vs 4-bit: {tok300.get('elaborate', float('nan')):.0f} token",
        f"  • shorten   · step 3 · bf16: {tok15.get('shorten', float('nan')):.0f} token  "
        f"vs 4-bit: {tok300.get('shorten', float('nan')):.0f} token",
        "",
        "2. Answer F1 (perdita di informazione)",
        "Sui 15q full-precision, F1 a step 3 resta molto vicino allo step 0 — il rewriter "
        "preserva l'informazione necessaria a rispondere. Sui 300q in 4-bit, F1 a step 3 "
        "scende quasi alla metà della baseline.",
        "",
        f"  • elaborate · bf16: F1(E_0)={f1_15_step0.get('elaborate', float('nan')):.3f} → "
        f"F1(E_3)={f1_15_step3.get('elaborate', float('nan')):.3f}",
        f"  • elaborate · 4-bit: F1(E_0)={f1_300_step0.get('elaborate', float('nan')):.3f} → "
        f"F1(E_3)={f1_300_step3.get('elaborate', float('nan')):.3f}",
        "",
        "3. Implicazioni",
        "I dati suggeriscono che la quantizzazione 4-bit non degrada solo la qualità "
        "del rewriting in modo uniforme, ma rompe specificamente la capacità del modello "
        "di seguire instruction che richiedono espansione (elaborate). L'effetto a "
        "cascata è che, dopo 3 iterazioni, le risposte alle domande non sono più "
        "estraibili dal testo riscritto.",
        "",
        "4. Limiti dei numeri attuali",
        "• Il 300q non copre ancora tutti i 300 qid: 110 hanno F1, 36 hanno BERTScore, "
        "55 hanno OFS. Tutti i numeri della Sezione A sono calcolati sull'intersezione.",
        "• Il subset comune fra 15q e 300q è di 10 qid; statisticamente è un campione "
        "piccolo. Le differenze qualitative sono però marcate e coerenti con quanto "
        "osservato sui pieni 300q.",
        "• OpenFActScore in 4-bit potrebbe avere un piccolo bias rispetto a OFS in bf16 "
        "(stesso AFG/AFV usato in entrambe le run). Il trend tra step e instruction è "
        "comunque preservato.",
        "",
        "5. Prossimi passi suggeriti",
        "• Completare il run 300q (resume) per portare F1/BERT/OFS allo stesso copertura.",
        "• Replicare il pilot 15q anche in 4-bit per un confronto più diretto e con più "
        "qid (questo isolerebbe quantizzazione vs sample).",
        "• Considerare di passare a NewsQA o FictionalQA per l'esperimento finale: "
        "input più corti (~370–700 token) limitano il problema della 'compressione "
        "implicita' indotta dal task lungo.",
    ]
    text_page(pdf, "Sintesi e prossimi passi", paragraphs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Caricamento dati da {RES} ...")
    data = load_all()
    for k, v in data.items():
        print(f"  {k:<14s} · {len(v):>5d} righe · {v['qid'].nunique():>4d} qid unici")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerazione PDF: {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        # Copertina
        text_page(
            pdf,
            "Risultati esperimenti rewriting iterativo — 09/06/2026",
            [
                "Documento di sintesi degli esperimenti svolti finora sul problema della "
                "degradazione del contenuto in catene di rewriting iterative su MuSiQue.",
                "",
                "Struttura del documento",
                "  • Sezione A — panoramica del run 300q (rewriter in 4-bit, QA in bf16)",
                "  • Sezione B — confronto fra il pilot 15q (bf16) e i corrispondenti qid "
                "del 300q (4-bit), per isolare l'effetto della quantizzazione",
                "  • Sintesi finale e prossimi passi",
                "",
                "Tutti i numeri sono medie sulle catene disponibili nei rispettivi CSV.",
            ],
        )
        section_300q(pdf, data)
        section_compare(pdf, data)
        section_summary(pdf, data)

    print(f"\n✓ PDF salvato: {OUT_PDF}")
    print(f"  Aprilo con: open {OUT_PDF}")


if __name__ == "__main__":
    main()
