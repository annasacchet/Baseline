"""
Heatmap Answer F1 per question x step (run 300q).

Tre pannelli affiancati, uno per livello di complessita (2-hop / 3-hop / 4-hop):
ogni cella = F1 medio della domanda allo step, mediato su istruzione e run.

Replica l'estetica di Figure 5 di visualize_trajectories_15q.py, adattata ai
~100 qid per hop del run 300q. Etichette qid solo a destra; ordinamento per
F1 medio decrescente dentro ciascun pannello (per leggibilita del gradiente).

Output: results/plots/300q/traj_f1_heatmap.pdf (+ png).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
F1_CSV = REPO_ROOT / "results" / "300q" / "rewriting_chains_300q_answer_f1_olmo31_4bit.csv"
OUT_DIR = REPO_ROOT / "results" / "plots" / "300q"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "png").mkdir(parents=True, exist_ok=True)

HOP_COLORS = {2: "#e15759", 3: "#f28e2b", 4: "#4e79a7"}
STEPS = [0, 1, 2, 3]


def hop_count(qid: str) -> int:
    m = re.match(r"(\d+)hop", qid)
    return int(m.group(1)) if m else 0


def short_label(qid: str) -> str:
    suffix = qid.split("__")[-1].split("_")[-1][-4:]
    return f"…{suffix}"


def panel(ax_strip, ax, pivot, hop, vmin=0.0, vmax=1.0):
    n = len(pivot)
    # strip colorata per hop (decorativa, omogenea perche un solo hop per pannello)
    strip_data = np.full((n, 1), hop, dtype=float)
    ax_strip.imshow(strip_data, aspect="auto",
                    cmap=plt.cm.colors.ListedColormap([HOP_COLORS[hop]]),
                    vmin=hop - 0.5, vmax=hop + 0.5, interpolation="nearest")
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    ax_strip.spines[:].set_visible(False)
    # testo nero se il colore della strip e' chiaro, bianco altrimenti
    r, g, b = plt.matplotlib.colors.to_rgb(HOP_COLORS[hop])
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    txt_color = "white" if lum < 0.55 else "black"
    ax_strip.text(0, n / 2, f"{hop}-hop  (n={n})",
                  ha="center", va="center", fontsize=10,
                  color=txt_color, fontweight="bold", rotation=90)

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(STEPS)))
    ax.set_xticklabels([f"Step {s}" for s in STEPS], fontsize=9)
    ax.set_yticks([])  # troppe etichette per ~100 qid; lasciamo pulito
    ax.set_yticklabels([])
    return im


def fig_heatmap(f1: pd.DataFrame) -> None:
    f1 = f1.copy()
    f1["hop"] = f1["qid"].apply(hop_count)

    # media su istruzione e run, qid x step
    full = (f1[f1["step"].isin(STEPS)]
            .groupby(["qid", "step"])["answer_f1"]
            .mean().unstack("step").reindex(columns=STEPS))
    full["hop"] = full.index.map(hop_count)

    panels = []
    for h in [2, 3, 4]:
        sub = full[full["hop"] == h].drop(columns="hop")
        sub["mean"] = sub[STEPS].mean(axis=1)
        sub = sub.sort_values("mean", ascending=False).drop(columns="mean")
        panels.append((h, sub))

    fig = plt.figure(figsize=(11, 8.5))
    ratios = []
    for _ in panels:
        ratios += [0.05, 1.0, 0.05]  # strip, heatmap, gap
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.02)

    im = None
    for i, (h, pivot) in enumerate(panels):
        ax_strip = fig.add_subplot(gs[0, i * 3])
        ax = fig.add_subplot(gs[0, i * 3 + 1])
        im = panel(ax_strip, ax, pivot, h)

    fig.suptitle("Answer F1 per question × step — 300q\n"
                 "(mean across instructions and runs; questions sorted by mean F1, desc)",
                 fontsize=12, fontweight="bold")
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Answer F1")
    fig.subplots_adjust(left=0.04, right=0.91, top=0.90, bottom=0.06)

    out_pdf = OUT_DIR / "traj_f1_heatmap.pdf"
    out_png = OUT_DIR / "png" / "traj_f1_heatmap.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


def main() -> None:
    f1 = pd.read_csv(F1_CSV)
    fig_heatmap(f1)


if __name__ == "__main__":
    main()
