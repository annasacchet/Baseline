"""
Grafici riassuntivi del pilot, pensati per la presentazione ai supervisor.

Output in slides/img/:
  - pilot_metrics_overview.png    : 4 metriche side-by-side (FS, BS, length, drop)
  - pilot_factscore_vs_length.png : scatter "perché shorten ha FS alto malgrado
                                    perda informazione" — la metrica nascosta
  - pilot_run_variability.png     : variabilità tra run su token counts
                                    (precursore P/A/U)

Pensati per:
  - leggibilità in proiezione (font grandi, pochi pannelli per figura)
  - raccontare il pilot, non per analisi esplorativa
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "slides" / "img"
OUT.mkdir(parents=True, exist_ok=True)

FS_BS = pd.read_csv(RESULTS / "rewriting_chains32b_factscore_bertscore.csv")

# Legacy CSVs (pre-Consecutive mode) used unsuffixed column names.
# Map them onto the current Baseline-mode names so older data still loads.
FS_BS = FS_BS.rename(columns={
    "bert_precision": "bert_precision_baseline",
    "bert_recall": "bert_recall_baseline",
    "bert_f1": "bert_f1_baseline",
})

TOK = pd.read_csv(RESULTS / "rewriting_chains32b_token_counts.csv")
AF1_PATH = RESULTS / "rewriting_chains32b_answer_f1 (1).csv"
AF1 = pd.read_csv(AF1_PATH) if AF1_PATH.exists() else None

COLORS = {
    "elaborate": "#e74c3c",
    "shorten": "#3498db",
    "formality": "#2ecc71",
    "paraphrase": "#f39c12",
}
MARKERS = {"elaborate": "o", "shorten": "s", "formality": "^", "paraphrase": "D"}
GROUP_OF = {
    "elaborate": "content",
    "shorten": "content",
    "formality": "style",
    "paraphrase": "style",
}

plt.rcParams.update({
    "font.size": 13,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# 1. Overview a 4 pannelli — la slide-cardine
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    "Pilot · 1 question 2-hop · run 0 · 4 instructions × 3 steps",
    fontsize=16, fontweight="bold", y=1.00,
)

# (a) FactScore per step
ax = axes[0, 0]
for itype in ["elaborate", "shorten", "formality", "paraphrase"]:
    sub = FS_BS[FS_BS["instruction_type"] == itype].sort_values("step")
    ax.plot(sub["step"], sub["factscore"], marker=MARKERS[itype], markersize=10,
            linewidth=2.5, color=COLORS[itype],
            label=f"{itype} ({GROUP_OF[itype]})")
ax.set_xticks([1, 2, 3])
ax.set_xlabel("Step")
ax.set_ylabel("FactScore (source-grounded)")
ax.set_title("(a) FactScore — factual fidelity vs E₀")
ax.set_ylim(0.80, 0.96)
ax.grid(alpha=0.3)
ax.legend(fontsize=10, loc="lower left")

# (b) BERTScore per step
ax = axes[0, 1]
for itype in ["elaborate", "shorten", "formality", "paraphrase"]:
    sub = FS_BS[FS_BS["instruction_type"] == itype].sort_values("step")
    ax.plot(sub["step"], sub["bert_f1_baseline"], marker=MARKERS[itype], markersize=10,
            linewidth=2.5, color=COLORS[itype],
            label=f"{itype} ({GROUP_OF[itype]})")
ax.set_xticks([1, 2, 3])
ax.set_xlabel("Step")
ax.set_ylabel("BERTScore F1")
ax.set_title("(b) BERTScore — semantic drift vs E₀")
ax.set_ylim(0.75, 0.96)
ax.grid(alpha=0.3)
ax.legend(fontsize=10, loc="lower left")

# (c) Token counts (run 0 — consistent with FS / BS in panels a, b, d)
ax = axes[1, 0]
tok_run0 = TOK[TOK["run"] == 0]
for itype in ["elaborate", "shorten", "formality", "paraphrase"]:
    sub = tok_run0[tok_run0["instruction_type"] == itype].sort_values("step")
    ax.plot(sub["step"], sub["n_tokens"], marker=MARKERS[itype], markersize=10,
            linewidth=2.5, color=COLORS[itype],
            label=f"{itype} ({GROUP_OF[itype]})")
ax.axhline(2357, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.text(0.05, 2400, "E₀ = 2357 tok", color="gray", fontsize=10)
ax.set_xticks([0, 1, 2, 3])
ax.set_xlabel("Step")
ax.set_ylabel("Token count (OLMo tokenizer)")
ax.set_title("(c) Length — run 0")
ax.grid(alpha=0.3)
ax.legend(fontsize=10, loc="upper right")

# (d) Answer F1 per step
ax = axes[1, 1]
if AF1 is not None:
    for itype in ["elaborate", "shorten", "formality", "paraphrase"]:
        sub = AF1[AF1["instruction_type"] == itype].sort_values("step")
        ax.plot(sub["step"], sub["answer_f1"], marker=MARKERS[itype],
                markersize=10, linewidth=2.5, color=COLORS[itype],
                label=f"{itype} ({GROUP_OF[itype]})")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Step")
    ax.set_ylabel("Answer F1 (vs MuSiQue gold)")
    ax.set_title("(d) Answer F1 — does the critical fact survive?")
    ax.set_ylim(0.55, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
else:
    ax.text(0.5, 0.5, "Answer F1 not available", ha="center", va="center")
    ax.axis("off")


plt.tight_layout()
plt.savefig(OUT / "pilot_metrics_overview.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"saved: {OUT / 'pilot_metrics_overview.png'}")


# ---------------------------------------------------------------------------
# 2. FactScore vs lunghezza — la slide che spiega il "trucco" di shorten
# ---------------------------------------------------------------------------

# Merge FS/BS con token counts per (qid, group, instr, run, step)
KEYS = ["qid", "group", "instruction_type", "run", "step"]
merged = FS_BS.merge(TOK[KEYS + ["n_tokens"]], on=KEYS, how="left")

fig, ax = plt.subplots(figsize=(11, 7))

for itype in ["elaborate", "shorten", "formality", "paraphrase"]:
    sub = merged[merged["instruction_type"] == itype].sort_values("step")
    ax.plot(sub["n_tokens"], sub["factscore"],
            color=COLORS[itype], linewidth=2, alpha=0.6, zorder=1)
    for _, r in sub.iterrows():
        ax.scatter(r["n_tokens"], r["factscore"],
                   marker=MARKERS[itype], s=200, color=COLORS[itype],
                   edgecolors="black", linewidth=1.3, zorder=3)
        ax.annotate(f"t{int(r['step'])}",
                    (r["n_tokens"], r["factscore"]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=10, fontweight="bold")

# legenda manuale
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker=MARKERS[i], color=COLORS[i], linewidth=2,
           markersize=12, markeredgecolor="black",
           label=f"{i} ({GROUP_OF[i]})")
    for i in ["elaborate", "shorten", "formality", "paraphrase"]
]
ax.legend(handles=handles, fontsize=12, loc="lower right")

ax.set_xlabel("Token count (length of Eₜ)", fontsize=13)
ax.set_ylabel("FactScore (source-grounded)", fontsize=13)
ax.set_title(
    "FactScore vs length: why 'shorten' stays high despite information loss\n"
    "(fewer facts → fewer chances to get them wrong, but the critical fact may disappear)",
    fontsize=13, pad=12,
)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "pilot_factscore_vs_length.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"saved: {OUT / 'pilot_factscore_vs_length.png'}")


# ---------------------------------------------------------------------------
# 3. Variabilità tra run — il segnale P/A/U che già abbiamo (sui token)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(
    "Cross-run variability (3 different prompt wordings) · token counts only (the only metric with 3 runs)",
    fontsize=14, fontweight="bold",
)

# (a) mean + min/max per instruction
ax = axes[0]
for itype in ["elaborate", "shorten", "formality", "paraphrase"]:
    sub = TOK[TOK["instruction_type"] == itype].groupby("step")["n_tokens"].agg(
        ["mean", "min", "max"]
    ).reset_index()
    ax.fill_between(sub["step"], sub["min"], sub["max"],
                    color=COLORS[itype], alpha=0.15)
    ax.plot(sub["step"], sub["mean"], marker=MARKERS[itype], markersize=10,
            linewidth=2.5, color=COLORS[itype],
            label=f"{itype} ({GROUP_OF[itype]})")
ax.set_xticks([0, 1, 2, 3])
ax.set_xlabel("Step")
ax.set_ylabel("Token count")
ax.set_title("(a) Mean and min–max band across 3 runs")
ax.grid(alpha=0.3)
ax.legend(fontsize=11, loc="upper right")

# (b) range (max-min) per instruction and step — how far apart the 3 runs are
ax = axes[1]
ranges = TOK.groupby(["instruction_type", "step"])["n_tokens"].agg(lambda x: x.max() - x.min()).reset_index()
ranges = ranges[ranges["step"] > 0]  # step 0 is the same for all runs
width = 0.2
steps = sorted(ranges["step"].unique())
itypes = ["elaborate", "shorten", "formality", "paraphrase"]
x = np.arange(len(steps))
for i, itype in enumerate(itypes):
    sub = ranges[ranges["instruction_type"] == itype].sort_values("step")
    offsets = x + (i - 1.5) * width
    ax.bar(offsets, sub["n_tokens"], width=width,
           color=COLORS[itype], edgecolor="black", linewidth=0.6,
           label=f"{itype} ({GROUP_OF[itype]})")
ax.set_xticks(x)
ax.set_xticklabels([f"t{s}" for s in steps])
ax.set_xlabel("Step")
ax.set_ylabel("Range (max − min) across 3 runs")
ax.set_title("(b) How much does the model vary across the 3 prompt wordings?")
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=10, loc="upper right")

plt.tight_layout()
plt.savefig(OUT / "pilot_run_variability.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"saved: {OUT / 'pilot_run_variability.png'}")

# ---------------------------------------------------------------------------
# 4. Composizione fatti — la metrica nascosta dentro FactScore aggregato
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=True)
fig.suptitle(
    "Fact composition per instruction · run 0 · 3 steps (t₁, t₂, t₃)\n"
    "Aggregate FactScore hides this decomposition",
    fontsize=14, fontweight="bold",
)

itypes_order = ["elaborate", "shorten", "formality", "paraphrase"]
for ax, itype in zip(axes, itypes_order):
    sub = FS_BS[FS_BS["instruction_type"] == itype].sort_values("step")
    steps = sub["step"].values
    n_facts = sub["n_facts"].values
    n_supp = sub["n_supported"].values
    n_contr = sub["n_contradicted"].values
    n_unsupp = n_facts - n_supp - n_contr

    width = 0.6
    p_supp = ax.bar(steps, n_supp, width, color="#2ecc71",
                    label="supported", edgecolor="black", linewidth=0.7)
    p_unsupp = ax.bar(steps, n_unsupp, width, bottom=n_supp,
                      color="#95a5a6", label="not supported",
                      edgecolor="black", linewidth=0.7)
    p_contr = ax.bar(steps, n_contr, width, bottom=n_supp + n_unsupp,
                     color="#e74c3c", label="contradicted",
                     edgecolor="black", linewidth=0.7)

    # total above each bar
    for i, s in enumerate(steps):
        ax.text(s, n_facts[i] + 5, f"n={int(n_facts[i])}",
                ha="center", fontsize=10, fontweight="bold")

    ax.set_title(f"{itype}\n({GROUP_OF[itype]})", fontsize=13)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["t₁", "t₂", "t₃"])
    ax.set_xlabel("step")
    if itype == "elaborate":
        ax.set_ylabel("Number of atomic facts")
    ax.grid(alpha=0.3, axis="y")

# legenda comune
axes[0].legend(loc="upper left", fontsize=10, framealpha=0.95)

plt.tight_layout()
plt.savefig(OUT / "pilot_fact_composition.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"saved: {OUT / 'pilot_fact_composition.png'}")


# ---------------------------------------------------------------------------
# 5. FS vs BS — scatter con traiettorie t1→t2→t3 per ogni chain
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 8.5))

for itype in ["elaborate", "shorten", "formality", "paraphrase"]:
    sub = FS_BS[FS_BS["instruction_type"] == itype].sort_values("step")
    fs = sub["factscore"].values
    bs = sub["bert_f1_baseline"].values
    # linea con frecce
    for i in range(len(fs) - 1):
        ax.annotate(
            "", xy=(fs[i + 1], bs[i + 1]), xytext=(fs[i], bs[i]),
            arrowprops=dict(arrowstyle="->", color=COLORS[itype],
                            lw=2.5, alpha=0.7),
        )
    ax.scatter(fs, bs, s=280, color=COLORS[itype], marker=MARKERS[itype],
               edgecolors="black", linewidth=1.3, zorder=5,
               label=f"{itype} ({GROUP_OF[itype]})")
    # etichette step
    for i, s in enumerate(sub["step"].values):
        ax.annotate(f"t{int(s)}", (fs[i], bs[i]),
                    xytext=(10, 8), textcoords="offset points",
                    fontsize=10, fontweight="bold")

# correlation
corr = FS_BS[["factscore", "bert_f1_baseline"]].corr().iloc[0, 1]
ax.set_xlabel("FactScore (source-grounded)", fontsize=14)
ax.set_ylabel("BERTScore F1", fontsize=14)
ax.set_title(
    f"Degradation trajectory in the FS × BS space\n"
    f"Pearson r = {corr:.3f} · arrows go from t₁ to t₃",
    fontsize=14, pad=12,
)
ax.grid(alpha=0.3)
ax.legend(fontsize=12, loc="lower right")

plt.tight_layout()
plt.savefig(OUT / "pilot_fs_bs_trajectory.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"saved: {OUT / 'pilot_fs_bs_trajectory.png'}")


# ---------------------------------------------------------------------------
# 6. Schema pipeline — diagramma a riquadri
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")

def box(x, y, w, h, text, color, fontsize=11, fontweight="bold", text_color="black"):
    rect = plt.Rectangle((x, y), w, h, linewidth=1.8, edgecolor="black",
                         facecolor=color, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=3)


def arrow(x1, y1, x2, y2, label=None, color="#2c3e50"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                mutation_scale=20),
                zorder=1)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1, label,
                ha="center", fontsize=9, style="italic", color=color)


# Title
ax.text(50, 57, "Current pipeline — pilot 1 question 2-hop",
        ha="center", fontsize=16, fontweight="bold")

# === STAGE 1: Dataset ===
box(2, 38, 16, 10, "MuSiQue\nans-dev\n(2417 q)", "#bdc3c7", fontsize=11)
ax.text(10, 35, "pilot qid:\n2hop__635544_110949",
        ha="center", fontsize=8, style="italic", color="#555")

# === STAGE 2: E_0 ===
box(24, 38, 14, 10, "E₀\n(evidence\nparagraphs)", "#fff3cd", fontsize=11)
ax.text(31, 34, "≈ 2357 tokens", ha="center", fontsize=8, style="italic")

# === STAGE 3: Rewriter ===
box(44, 38, 18, 10, "OLMo-3 32B\n(rewriter)", "#3498db", fontsize=11, text_color="white")
ax.text(53, 34, "4 instr. × 3 runs × 3 steps", ha="center", fontsize=8,
        style="italic", color="#555")

# === STAGE 4: Chain ===
box(68, 38, 22, 10, "Eₜ chain\nE₀ → E₁ → E₂ → E₃",
    "#fff3cd", fontsize=11)
ax.text(79, 34, "48 texts total", ha="center", fontsize=8, style="italic")

# Horizontal arrows stage 1→4
arrow(18, 43, 24, 43)
arrow(38, 43, 44, 43)
arrow(62, 43, 68, 43)

# === Branching: 4 metrics ===
metric_y = 14
metric_h = 10

# FactScore
box(2, metric_y, 19, metric_h,
    "FactScore\n(judge: gpt-4o-mini)\nfactual fidelity vs E₀",
    "#e74c3c", fontsize=10, text_color="white")
ax.text(11.5, metric_y - 3, "1 run · steps 1-3", ha="center", fontsize=8,
        style="italic", color="#555")

# BERTScore
box(23, metric_y, 19, metric_h,
    "BERTScore\n(roberta-large)\nsemantic drift vs E₀",
    "#9b59b6", fontsize=10, text_color="white")
ax.text(32.5, metric_y - 3, "1 run · steps 1-3", ha="center", fontsize=8,
        style="italic", color="#555")

# Token counts
box(44, metric_y, 19, metric_h,
    "Token counts\n(OLMo tokenizer)\nlength of Eₜ",
    "#2ecc71", fontsize=10, text_color="white")
ax.text(53.5, metric_y - 3, "3 runs · steps 0-3", ha="center", fontsize=8,
        style="italic", color="#555")

# Answer F1
box(65, metric_y, 19, metric_h,
    "Answer F1\n(QA: OLMo-32B-Instruct)\ncritical fact in Eₜ",
    "#f39c12", fontsize=10, text_color="white")
ax.text(74.5, metric_y - 3, "1 run · steps 0-3", ha="center", fontsize=8,
        style="italic", color="#555")

# Arrows from the chain to the metrics
for x_target in [11.5, 32.5, 53.5, 74.5]:
    arrow(79, 38, x_target, metric_y + metric_h, color="#7f8c8d")

plt.tight_layout()
plt.savefig(OUT / "pipeline_overview.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"saved: {OUT / 'pipeline_overview.png'}")

# ---------------------------------------------------------------------------
# 7. Answer F1 — risultati pilot
# ---------------------------------------------------------------------------

if AF1 is not None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Answer F1 vs FactScore — comparison on run 0",
                 fontsize=14, fontweight="bold")

    # (a) Answer F1 per step
    ax = axes[0]
    for itype in ["elaborate", "shorten", "formality", "paraphrase"]:
        sub = AF1[AF1["instruction_type"] == itype].sort_values("step")
        ax.plot(sub["step"], sub["answer_f1"], marker=MARKERS[itype],
                markersize=10, linewidth=2.5, color=COLORS[itype],
                label=f"{itype} ({GROUP_OF[itype]})")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Step")
    ax.set_ylabel("Answer F1 (vs MuSiQue gold)")
    ax.set_title("(a) Answer F1 per step")
    ax.set_ylim(0.55, 1.05)
    ax.axhline(0.75, color="gray", linestyle=":", alpha=0.6, linewidth=1)
    ax.text(0.05, 0.77, "F1 baseline on E₀ = 0.75", color="gray", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")

    # (b) FactScore vs Answer F1 — trend comparison
    ax = axes[1]
    fs_run0 = FS_BS[FS_BS["run"] == 0]
    for itype in ["elaborate", "shorten", "formality", "paraphrase"]:
        fs_sub = fs_run0[fs_run0["instruction_type"] == itype].sort_values("step")
        af_sub = AF1[(AF1["instruction_type"] == itype) & (AF1["step"] > 0)].sort_values("step")
        ax.plot(fs_sub["step"], fs_sub["factscore"],
                marker=MARKERS[itype], markersize=8, linewidth=2,
                color=COLORS[itype], alpha=0.8,
                label=f"{itype}: FactScore")
        ax.plot(af_sub["step"], af_sub["answer_f1"],
                marker=MARKERS[itype], markersize=8, linewidth=2,
                color=COLORS[itype], alpha=0.8, linestyle="--",
                label=f"{itype}: Answer F1")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.set_title("(b) FactScore (solid) vs Answer F1 (dashed)")
    ax.set_ylim(0.55, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", ncol=2)

    plt.tight_layout()
    plt.savefig(OUT / "pilot_answer_f1.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"saved: {OUT / 'pilot_answer_f1.png'}")


print("\nDone. 7 charts in slides/img/:")
print("  - pipeline_overview.png            (pipeline schema)")
print("  - pilot_metrics_overview.png       (4-metric overview)")
print("  - pilot_factscore_vs_length.png    (FS vs length scatter)")
print("  - pilot_run_variability.png        (cross-run variability on tokens)")
print("  - pilot_fact_composition.png       (fact composition)")
print("  - pilot_fs_bs_trajectory.png       (FS vs BS trajectories)")
print("  - pilot_answer_f1.png              (Answer F1 results)")
