"""
Visualizzazione BERTScore per il pilota 15q.

Plotta entrambe le modalità di §6.2:
  - Baseline:    sim(E_t, E_0)   — drift cumulativo dall'originale
  - Consecutive: sim(E_t, E_{t-1}) — cambiamento tra step adiacenti

Aggrega le 45 chain per (instruction_type, step) con mediana + banda IQR.

Output:
  - bertscore_15q_by_instruction.pdf  — un pannello per istruzione (4)
                                         con Baseline + Consecutive sovrapposte
  - bertscore_15q_summary.pdf         — overlay delle 4 istruzioni, due pannelli
                                         (uno Baseline, uno Consecutive)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q_bertscore.csv"
OUT_BY_INSTR = REPO_ROOT / "results" / "plots" / "15q" / "bertscore_15q_by_instruction.pdf"
OUT_SUMMARY = REPO_ROOT / "results" / "plots" / "15q" / "bertscore_15q_summary.pdf"

COLORS = {
    "elaborate": "#e74c3c",
    "shorten": "#3498db",
    "formality": "#2ecc71",
    "paraphrase": "#f39c12",
}
MARKERS = {
    "elaborate": "o",
    "shorten": "s",
    "formality": "^",
    "paraphrase": "D",
}
GROUP_OF = {
    "elaborate": "content",
    "shorten": "content",
    "formality": "style",
    "paraphrase": "style",
}


def summarize(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mediana + IQR per (instruction_type, step) sulla colonna data."""
    return (
        df.groupby(["instruction_type", "step"])[value_col]
        .agg(
            median="median",
            mean="mean",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            n="count",
        )
        .reset_index()
    )


def plot_by_instruction(df: pd.DataFrame, out_path: Path):
    """Un pannello per istruzione, due linee (Baseline + Consecutive) per pannello."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    summary_b = summarize(df, "bert_f1_baseline")
    summary_c = summarize(df, "bert_f1_consecutive")

    for i, itype in enumerate(["elaborate", "shorten", "formality", "paraphrase"]):
        ax = axes[i]
        sub_b = summary_b[summary_b["instruction_type"] == itype].sort_values("step")
        sub_c = summary_c[summary_c["instruction_type"] == itype].sort_values("step")

        # Baseline (vs E_0): linea solida + banda IQR
        ax.fill_between(
            sub_b["step"], sub_b["q25"], sub_b["q75"],
            color=COLORS[itype], alpha=0.18,
        )
        ax.plot(
            sub_b["step"], sub_b["median"],
            marker=MARKERS[itype], markersize=10, linewidth=2.5,
            color=COLORS[itype], label="Baseline (vs E₀)",
        )

        # Consecutive (vs E_{t-1}): linea tratteggiata + banda IQR
        ax.fill_between(
            sub_c["step"], sub_c["q25"], sub_c["q75"],
            color=COLORS[itype], alpha=0.10, hatch="//", edgecolor=COLORS[itype],
        )
        ax.plot(
            sub_c["step"], sub_c["median"],
            marker=MARKERS[itype], markersize=10, linewidth=2.5,
            color=COLORS[itype], linestyle="--", alpha=0.7,
            label="Consecutive (vs E_{t-1})",
        )

        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Step")
        ax.set_ylabel("BERTScore F1")
        ax.set_title(f"{itype.capitalize()} ({GROUP_OF[itype]})")
        ax.set_ylim(0.78, 1.005)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "BERTScore per step — pilot 15q (5 questions × 3 hop)\n"
        "Median across 45 chains per cell · IQR band",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def plot_summary(df: pd.DataFrame, out_path: Path):
    """Due pannelli affiancati: Baseline a sx, Consecutive a dx, 4 istruzioni overlaid."""
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(15, 6))

    summary_b = summarize(df, "bert_f1_baseline")
    summary_c = summarize(df, "bert_f1_consecutive")

    for itype in ["elaborate", "shorten", "formality", "paraphrase"]:
        sub_b = summary_b[summary_b["instruction_type"] == itype].sort_values("step")
        sub_c = summary_c[summary_c["instruction_type"] == itype].sort_values("step")

        ax_b.plot(
            sub_b["step"], sub_b["median"],
            marker=MARKERS[itype], markersize=10, linewidth=2.5,
            color=COLORS[itype],
            label=f"{itype} ({GROUP_OF[itype]})",
        )
        ax_c.plot(
            sub_c["step"], sub_c["median"],
            marker=MARKERS[itype], markersize=10, linewidth=2.5,
            color=COLORS[itype],
            label=f"{itype} ({GROUP_OF[itype]})",
        )

    for ax, title, ylim in [
        (ax_b, "Baseline — sim(E_t, E_0)\ncumulative drift from original", (0.80, 0.95)),
        (ax_c, "Consecutive — sim(E_t, E_{t-1})\nchange between adjacent steps", (0.80, 1.00)),
    ]:
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("BERTScore F1 (median)", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)

    fig.suptitle(
        "BERTScore — pilot 15q · median across 45 chains per (instruction, step)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def main():
    df = pd.read_csv(CSV_PATH)

    print("=" * 70)
    print("BERTScore — pilot 15q (5 questions × 3 hop)")
    print("=" * 70)
    print(f"Total rows: {len(df)}  |  unique chains: {df.groupby(['qid','group','instruction_type','run']).ngroups}")
    print()

    summary_b = summarize(df, "bert_f1_baseline")
    summary_c = summarize(df, "bert_f1_consecutive")

    print("Baseline F1 — median per (instruction, step):")
    print(summary_b.pivot(index="instruction_type", columns="step", values="median").round(3))
    print()
    print("Consecutive F1 — median per (instruction, step):")
    print(summary_c.pivot(index="instruction_type", columns="step", values="median").round(3))
    print()

    plot_by_instruction(df, OUT_BY_INSTR)
    plot_summary(df, OUT_SUMMARY)


if __name__ == "__main__":
    main()
