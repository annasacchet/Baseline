"""
Trajectory plots — answerable chains only (F1 > 0 at step 1).

Figure 1 — F1 by instruction type (mean + band)
Figure 2 — BERTScore by instruction type (mean + band)
Figure 3 — F1 by hop count (mean + band)
Figure 4 — F1 and BERTScore side by side by instruction type
Figure 5 — F1 heatmap per question x step
"""

import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
F1_CSV  = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q_answer_f1.csv"
BS_CSV  = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q_bertscore.csv"
OUT_DIR = REPO_ROOT / "results" / "plots" / "15q"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTIONS = ["formality", "paraphrase", "shorten", "elaborate"]
COLORS = {
    "formality":  "#4e79a7",
    "paraphrase": "#f28e2b",
    "shorten":    "#e15759",
    "elaborate":  "#59a14f",
}
HOP_COLORS = {2: "#e15759", 3: "#f28e2b", 4: "#4e79a7"}


def hop_count(qid):
    m = re.match(r"(\d+)hop", qid)
    return int(m.group(1)) if m else 0


def load():
    f1 = pd.read_csv(F1_CSV)
    bs = pd.read_csv(BS_CSV)
    f1["chain_id"] = f1["qid"] + "|" + f1["instruction_type"] + "|" + f1["run"].astype(str)
    f1["hop"] = f1["qid"].apply(hop_count)

    # filter: only chains answerable at step 1
    answerable = f1[f1["step"] == 1].query("answer_f1 > 0")["chain_id"].unique()
    f1 = f1[f1["chain_id"].isin(answerable)].copy()

    bs["chain_id"] = bs["qid"] + "|" + bs["instruction_type"] + "|" + bs["run"].astype(str)
    bs = bs[bs["chain_id"].isin(answerable)].copy()

    print(f"Answerable chains: {len(answerable)}/180  |  questions: {f1['qid'].nunique()}/15")
    return f1, bs


def style_ax(ax, title, xlabel, ylabel, xlim=None, ylim=None):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(labelsize=9)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def mean_band(ax, df, steps, metric, color, label):
    means = df.groupby("step")[metric].mean().reindex(steps)
    stds  = df.groupby("step")[metric].std().reindex(steps)
    ax.plot(steps, means.values, color=color, linewidth=2.5,
            marker="o", markersize=7, markerfacecolor="white",
            markeredgewidth=2, label=label, zorder=3)
    ax.fill_between(steps,
                    (means - stds).clip(lower=0).values,
                    (means + stds).clip(upper=1).values,
                    color=color, alpha=0.12, zorder=1)
    # annotate every point
    for s, v in zip(steps, means.values):
        if not np.isnan(v):
            ax.text(s, v + 0.025, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="bold", zorder=4)
    return means


# ---------------------------------------------------------------------------
# Figure 1 — F1 by instruction (mean + band, single panel)
# ---------------------------------------------------------------------------

def fig1_f1_by_instruction(f1):
    fig, ax = plt.subplots(figsize=(7, 5))
    steps = [1, 2, 3]

    for instr in INSTRUCTIONS:
        sub = f1[(f1["instruction_type"] == instr) & (f1["step"].isin(steps))]
        mean_band(ax, sub, steps, "answer_f1", COLORS[instr], instr)

    ax.set_xticks(steps)
    ax.set_xticklabels(["Step 1\n(1st rewrite)", "Step 2", "Step 3\n(3rd rewrite)"], fontsize=9)
    style_ax(ax, "Answer F1 across rewriting steps",
             "", "Answer F1", xlim=(0.7, 3.5), ylim=(0, 1.05))
    ax.legend(title="Instruction", fontsize=9, title_fontsize=9,
              loc="upper left", framealpha=0.9)
    n = f1[f1["step"] == 1]["chain_id"].nunique()
    ax.text(0.02, 0.02, f"n = {n} answerable chains",
            transform=ax.transAxes, fontsize=8, color="#777")
    fig.tight_layout()
    out = OUT_DIR / "traj_f1_by_instruction.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2 — BERTScore by instruction (mean + band, single panel)
# ---------------------------------------------------------------------------

def fig2_bert_by_instruction(bs):
    fig, ax = plt.subplots(figsize=(7, 5))
    steps = [1, 2, 3]

    for instr in INSTRUCTIONS:
        sub = bs[(bs["instruction_type"] == instr) & (bs["step"].isin(steps))]
        mean_band(ax, sub, steps, "bert_f1_baseline", COLORS[instr], instr)

    ax.set_xticks(steps)
    ax.set_xticklabels(["Step 1\n(1st rewrite)", "Step 2", "Step 3\n(3rd rewrite)"], fontsize=9)
    style_ax(ax, "Semantic drift from E₀ (BERTScore)",
             "", "BERTScore F1 vs E₀", xlim=(0.7, 3.5), ylim=(0.75, 1.02))
    ax.legend(title="Instruction", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.9)
    n = bs["chain_id"].nunique()
    ax.text(0.02, 0.02, f"n = {n} answerable chains",
            transform=ax.transAxes, fontsize=8, color="#777")
    fig.tight_layout()
    out = OUT_DIR / "traj_bert_by_instruction.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3 — F1 by hop count (3 lines, single panel)
# ---------------------------------------------------------------------------

def fig3_f1_by_hop(f1):
    fig, ax = plt.subplots(figsize=(7, 5))
    steps = [1, 2, 3]

    for hop in [2, 3, 4]:
        sub = f1[(f1["hop"] == hop) & (f1["step"].isin(steps))]
        n = sub[sub["step"] == 1]["chain_id"].nunique()
        mean_band(ax, sub, steps, "answer_f1",
                  HOP_COLORS[hop], f"{hop}-hop  (n={n})")

    ax.set_xticks(steps)
    ax.set_xticklabels(["Step 1\n(1st rewrite)", "Step 2", "Step 3\n(3rd rewrite)"], fontsize=9)
    style_ax(ax, "Answer F1 by question complexity (hop count)",
             "", "Answer F1", xlim=(0.7, 3.5), ylim=(0, 1.05))
    ax.legend(title="Hop count", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out = OUT_DIR / "traj_f1_by_hop.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4 — F1 and BERTScore side by side (left=F1, right=BERTScore)
# One line per instruction, mean + band, same structure both panels
# ---------------------------------------------------------------------------

def fig4_f1_vs_bert(f1, bs):
    steps = [1, 2, 3]
    fig, (ax_f1, ax_bs) = plt.subplots(1, 2, figsize=(12, 5))

    for instr in INSTRUCTIONS:
        color = COLORS[instr]

        # left: F1
        sub_f1 = f1[(f1["instruction_type"] == instr) & (f1["step"].isin(steps))]
        mean_band(ax_f1, sub_f1, steps, "answer_f1", color, instr)

        # right: BERTScore
        sub_bs = bs[(bs["instruction_type"] == instr) & (bs["step"].isin(steps))]
        mean_band(ax_bs, sub_bs, steps, "bert_f1_baseline", color, instr)

    for ax, title, ylabel, ylim in [
        (ax_f1, "Answer F1", "Answer F1", (0.0, 1.05)),
        (ax_bs, "Semantic drift from E₀ (BERTScore)", "BERTScore F1 vs E₀", (0.75, 1.02)),
    ]:
        ax.set_xticks(steps)
        ax.set_xticklabels(["Step 1\n(1st rewrite)", "Step 2", "Step 3\n(3rd rewrite)"], fontsize=9)
        style_ax(ax, title, "", ylabel, xlim=(0.7, 3.5), ylim=ylim)

    ax_f1.legend(title="Instruction", fontsize=9, title_fontsize=9,
                 loc="lower left", framealpha=0.9)

    n = f1[f1["step"] == 1]["chain_id"].nunique()
    fig.text(0.5, -0.02, f"n = {n} answerable chains  (mean ± 1 std)",
             ha="center", fontsize=9, color="#777")
    fig.suptitle("Answer F1 and BERTScore across rewriting steps",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "traj_f1_vs_bert.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 5 — F1 heatmap: question x step, mean across all instructions+runs
# One panel only, ordered by hop count, no NaN rows
# ---------------------------------------------------------------------------

def fig5_heatmap(f1):
    steps = [1, 2, 3]

    def hop_from_qid(qid):
        m = re.match(r"(\d+)hop", qid)
        return int(m.group(1)) if m else 0

    # mean across all instructions and runs
    pivot = (f1[f1["step"].isin(steps)]
             .groupby(["qid", "step"])["answer_f1"]
             .mean()
             .unstack("step")
             .reindex(columns=steps))

    # add hop, sort by hop then by mean F1 descending within hop
    pivot["hop"] = pivot.index.map(hop_from_qid)
    pivot["mean_f1"] = pivot[[1, 2, 3]].mean(axis=1)
    pivot = pivot.sort_values(["hop", "mean_f1"], ascending=[True, False])
    hops = pivot.pop("hop")
    pivot.pop("mean_f1")

    # readable labels: "2-hop · ...8311"
    def label(qid):
        h = hop_from_qid(qid)
        suffix = qid.split("__")[-1].split("_")[-1][-4:]
        return f"{h}-hop · …{suffix}"

    labels = [label(q) for q in pivot.index]

    hop_vals = [hop_from_qid(q) for q in pivot.index]

    # coloured strip on the left (one thin axes)
    fig, (ax_strip, ax) = plt.subplots(1, 2, figsize=(6, 5.5),
                                        gridspec_kw={"width_ratios": [0.08, 1]})
    strip_colors = np.array([[HOP_COLORS[h][1:]] for h in hop_vals], dtype=object)
    strip_data = np.array([[h] for h in hop_vals], dtype=float)
    ax_strip.imshow(strip_data, aspect="auto",
                    cmap=plt.cm.colors.ListedColormap([HOP_COLORS[h] for h in sorted(set(hop_vals))]),
                    vmin=min(hop_vals) - 0.5, vmax=max(hop_vals) + 0.5,
                    interpolation="nearest")
    ax_strip.set_xticks([])
    ax_strip.set_yticks(range(len(hop_vals)))
    ax_strip.set_yticklabels([])
    ax_strip.spines[:].set_visible(False)

    # hop group labels centred on the strip
    seen = {}
    for i, h in enumerate(hop_vals):
        seen.setdefault(h, []).append(i)
    for h, idxs in seen.items():
        mid = np.mean(idxs)
        ax_strip.text(0, mid, f"{h}-hop", ha="center", va="center",
                      fontsize=7.5, color="white", fontweight="bold", rotation=90)

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(["Step 1", "Step 2", "Step 3"], fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.yaxis.tick_right()

    # hop group dividers
    for i in range(1, len(hop_vals)):
        if hop_vals[i] != hop_vals[i - 1]:
            ax.axhline(i - 0.5, color="white", linewidth=2)
            ax_strip.axhline(i - 0.5, color="white", linewidth=2)

    # cell values
    for i in range(len(pivot)):
        for j in range(len(steps)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                txt_color = "white" if v < 0.35 or v > 0.75 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Answer F1", pad=0.02)
    ax.set_title("Answer F1 per question\n(mean across instructions and runs)",
                 fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    out = OUT_DIR / "traj_f1_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 6 — Dot plot: one line per question, F1 at step 1→3
# coloured by hop count
# ---------------------------------------------------------------------------

def fig6_dotplot(f1):
    steps = [1, 2, 3]

    def hop_from_qid(qid):
        m = re.match(r"(\d+)hop", qid)
        return int(m.group(1)) if m else 0

    # mean across instructions and runs per (qid, step)
    agg = (f1[f1["step"].isin(steps)]
           .groupby(["qid", "step"])["answer_f1"]
           .mean()
           .reset_index())
    agg["hop"] = agg["qid"].apply(hop_from_qid)

    # sort questions: by hop, then by F1 at step 1 descending
    order = (agg[agg["step"] == 1]
             .sort_values(["hop", "answer_f1"], ascending=[True, False])["qid"]
             .tolist())

    def label(qid):
        h = hop_from_qid(qid)
        suffix = qid.split("__")[-1].split("_")[-1][-4:]
        return f"{h}-hop · …{suffix}"

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for qid in order:
        sub = agg[agg["qid"] == qid].sort_values("step")
        hop = sub["hop"].iloc[0]
        color = HOP_COLORS[hop]
        ys = sub["answer_f1"].values
        xs = sub["step"].values
        ax.plot(xs, ys, color=color, linewidth=1.4, alpha=0.7, zorder=2)
        ax.scatter(xs, ys, color=color, s=40, zorder=3, alpha=0.85, edgecolors="none")
        # label at step 3
        ax.text(3.06, float(ys[-1]), label(qid),
                va="center", fontsize=7.5, color=color)

    ax.set_xticks(steps)
    ax.set_xticklabels(["Step 1\n(1st rewrite)", "Step 2", "Step 3\n(3rd rewrite)"], fontsize=9)
    ax.set_xlim(0.7, 4.0)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Answer F1", fontsize=10)
    ax.set_title("Answer F1 trajectory per question\n(mean across instructions and runs)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(labelsize=9)

    legend_els = [mpatches.Patch(color=HOP_COLORS[h], label=f"{h}-hop") for h in [2, 3, 4]]
    ax.legend(handles=legend_els, title="Hop count", fontsize=9,
              title_fontsize=9, loc="lower left", framealpha=0.9)

    fig.tight_layout()
    out = OUT_DIR / "traj_f1_dotplot.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 7 — Style vs Content: two lines, F1 and BERTScore side by side
# ---------------------------------------------------------------------------

def fig7_style_vs_content(f1, bs):
    steps = [1, 2, 3]
    GROUP_COLORS = {"style": "#4e79a7", "content": "#e15759"}
    GROUP_LABELS = {"style": "Style-oriented\n(formality, paraphrase)",
                    "content": "Content-oriented\n(shorten, elaborate)"}

    fig, (ax_f1, ax_bs) = plt.subplots(1, 2, figsize=(12, 5))

    for group in ["style", "content"]:
        color = GROUP_COLORS[group]
        label = GROUP_LABELS[group]

        # F1
        sub_f1 = f1[(f1["group"] == group) & (f1["step"].isin(steps))]
        mean_band(ax_f1, sub_f1, steps, "answer_f1", color, label)

        # BERTScore
        sub_bs = bs[(bs["group"] == group) & (bs["step"].isin(steps))]
        mean_band(ax_bs, sub_bs, steps, "bert_f1_baseline", color, label)

    for ax, title, ylabel, ylim in [
        (ax_f1, "Answer F1", "Answer F1", (0.0, 1.05)),
        (ax_bs, "Semantic drift from E₀ (BERTScore)", "BERTScore F1 vs E₀", (0.75, 1.02)),
    ]:
        ax.set_xticks(steps)
        ax.set_xticklabels(["Step 1\n(1st rewrite)", "Step 2", "Step 3\n(3rd rewrite)"], fontsize=9)
        style_ax(ax, title, "", ylabel, xlim=(0.7, 3.5), ylim=ylim)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.9)

    n = f1[f1["step"] == 1]["chain_id"].nunique()
    fig.text(0.5, -0.02, f"n = {n} answerable chains  (mean ± 1 std)",
             ha="center", fontsize=9, color="#777")
    fig.suptitle("Style-oriented vs Content-oriented instructions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "traj_style_vs_content.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    f1, bs = load()

    print("\nFigure 1: F1 by instruction...")
    fig1_f1_by_instruction(f1)

    print("Figure 2: BERTScore by instruction...")
    fig2_bert_by_instruction(bs)

    print("Figure 3: F1 by hop count...")
    fig3_f1_by_hop(f1)

    print("Figure 4: F1 vs BERTScore...")
    fig4_f1_vs_bert(f1, bs)

    print("Figure 5: F1 heatmap per question...")
    fig5_heatmap(f1)

    print("Figure 6: F1 dot plot per question...")
    fig6_dotplot(f1)

    print("Figure 7: Style vs Content...")
    fig7_style_vs_content(f1, bs)

    print(f"\nDone. Plots saved to: {OUT_DIR}")
