"""
Plots for answerable (F1>0 at step0) vs non-answerable (F1=0 at step0) chains.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="15q", help="Dataset tag (e.g. 15q, 200q).")
    return parser.parse_args()


_args   = parse_args()
TAG     = _args.tag
F1_CSV  = REPO_ROOT / "results" / TAG / f"rewriting_chains_{TAG}_answer_f1.csv"
OFS_CSV = REPO_ROOT / "results" / TAG / f"rewriting_chains_{TAG}_openfactscore.csv"
BS_CSV  = REPO_ROOT / "results" / TAG / f"rewriting_chains_{TAG}_bertscore.csv"
OUT_DIR = REPO_ROOT / "results" / "plots" / TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "png").mkdir(parents=True, exist_ok=True)

COLORS = {"answerable": "#2196F3", "non-answerable": "#F44336"}
INSTRUCTIONS = ["elaborate", "shorten", "paraphrase", "formality"]
STEPS = [0, 1, 2, 3]


def load_data():
    f1  = pd.read_csv(F1_CSV)
    ofs = pd.read_csv(OFS_CSV)
    bs  = pd.read_csv(BS_CSV)

    # Answerable flag from step 0
    f1_0 = f1[f1["step"] == 0][["qid", "instruction_type", "run", "answer_f1"]].copy()
    f1_0["answerable"] = f1_0["answer_f1"] > 0

    f1  = f1.merge(f1_0[["qid","instruction_type","run","answerable"]], on=["qid","instruction_type","run"])
    ofs = ofs.merge(f1_0[["qid","instruction_type","run","answerable"]], on=["qid","instruction_type","run"])
    bs  = bs.merge(f1_0[["qid","instruction_type","run","answerable"]], on=["qid","instruction_type","run"])

    f1["hop"]  = f1["qid"].str.extract(r"^(\d+)hop").astype(int)
    ofs["hop"] = ofs["qid"].str.extract(r"^(\d+)hop").astype(int)
    bs["hop"]  = bs["qid"].str.extract(r"^(\d+)hop").astype(int)

    return f1, ofs, bs


def plot_trajectory(ax, df, metric, steps, group_col, group_vals, labels, colors, title, ylabel):
    for val, label, color in zip(group_vals, labels, colors):
        sub = df[df[group_col] == val]
        means = sub.groupby("step")[metric].mean()
        sems  = sub.groupby("step")[metric].sem()
        y = [means.get(s, np.nan) for s in steps]
        e = [sems.get(s, np.nan)  for s in steps]
        ax.plot(steps, y, marker="o", color=color, label=label, linewidth=2)
        ax.fill_between(steps,
                         [a - b for a,b in zip(y,e)],
                         [a + b for a,b in zip(y,e)],
                         color=color, alpha=0.15)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Step")
    ax.set_xticks(steps)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


def main():
    f1, ofs, bs = load_data()

    ans_labels = ["answerable", "non-answerable"]
    ans_vals   = [True, False]
    ans_colors = [COLORS["answerable"], COLORS["non-answerable"]]

    # ------------------------------------------------------------------
    # Fig 1: Answer F1 trajectory — answerable vs non-answerable, by instruction
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    fig.suptitle("Answer F1 trajectory: answerable vs non-answerable", fontsize=13)
    for ax, instr in zip(axes.flat, INSTRUCTIONS):
        sub = f1[f1["instruction_type"] == instr]
        plot_trajectory(ax, sub, "answer_f1", STEPS, "answerable",
                        ans_vals, ans_labels, ans_colors,
                        instr, "Answer F1")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "answerability_f1_by_instruction.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "png" / "answerability_f1_by_instruction.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: answerability_f1_by_instruction")

    # ------------------------------------------------------------------
    # Fig 2: FactScore trajectory — answerable vs non-answerable, by instruction
    ofs_steps = [1, 2, 3]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    fig.suptitle("FactScore trajectory: answerable vs non-answerable", fontsize=13)
    for ax, instr in zip(axes.flat, INSTRUCTIONS):
        sub = ofs[ofs["instruction_type"] == instr]
        plot_trajectory(ax, sub, "factscore", ofs_steps, "answerable",
                        ans_vals, ans_labels, ans_colors,
                        instr, "FactScore")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "answerability_factscore_by_instruction.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "png" / "answerability_factscore_by_instruction.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: answerability_factscore_by_instruction")

    # ------------------------------------------------------------------
    # Fig 3: BERTScore baseline trajectory — answerable vs non-answerable, by instruction
    bs_steps = [1, 2, 3]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    fig.suptitle("BERTScore (baseline) trajectory: answerable vs non-answerable", fontsize=13)
    for ax, instr in zip(axes.flat, INSTRUCTIONS):
        sub = bs[bs["instruction_type"] == instr]
        plot_trajectory(ax, sub, "bert_f1_baseline", bs_steps, "answerable",
                        ans_vals, ans_labels, ans_colors,
                        instr, "BERTScore F1 (baseline)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "answerability_bertscore_by_instruction.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "png" / "answerability_bertscore_by_instruction.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: answerability_bertscore_by_instruction")

    # ------------------------------------------------------------------
    # Fig 4: F1 distribution at step0 — boxplot by hop count
    fig, ax = plt.subplots(figsize=(7, 5))
    f1_step0 = f1[f1["step"] == 0]
    hop_vals = sorted(f1_step0["hop"].unique())
    data_ans     = [f1_step0[(f1_step0["hop"]==h) & (f1_step0["answerable"]==True)]["answer_f1"].values  for h in hop_vals]
    data_nonans  = [f1_step0[(f1_step0["hop"]==h) & (f1_step0["answerable"]==False)]["answer_f1"].values for h in hop_vals]
    x = np.arange(len(hop_vals))
    bp1 = ax.boxplot(data_ans,    positions=x - 0.2, widths=0.35, patch_artist=True,
                     boxprops=dict(facecolor=COLORS["answerable"], alpha=0.7))
    bp2 = ax.boxplot(data_nonans, positions=x + 0.2, widths=0.35, patch_artist=True,
                     boxprops=dict(facecolor=COLORS["non-answerable"], alpha=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}-hop" for h in hop_vals])
    ax.set_ylabel("Answer F1 at step 0")
    ax.set_title("Answer F1 at step 0 by hop count and answerability")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["answerable", "non-answerable"])
    plt.tight_layout()
    fig.savefig(OUT_DIR / "answerability_f1_step0_by_hop.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "png" / "answerability_f1_step0_by_hop.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: answerability_f1_step0_by_hop")

    # ------------------------------------------------------------------
    # Fig 5: FactScore step1 — answerable vs non-answerable, all instructions combined
    fig, ax = plt.subplots(figsize=(7, 5))
    for val, label, color in zip(ans_vals, ans_labels, ans_colors):
        sub = ofs[ofs["answerable"] == val]
        means = sub.groupby("step")["factscore"].mean()
        sems  = sub.groupby("step")["factscore"].sem()
        y = [means.get(s, np.nan) for s in ofs_steps]
        e = [sems.get(s, np.nan)  for s in ofs_steps]
        ax.plot(ofs_steps, y, marker="o", color=color, label=label, linewidth=2)
        ax.fill_between(ofs_steps, [a-b for a,b in zip(y,e)], [a+b for a,b in zip(y,e)],
                        color=color, alpha=0.15)
    ax.set_title("FactScore trajectory (all instructions): answerable vs non-answerable")
    ax.set_ylabel("FactScore")
    ax.set_xlabel("Step")
    ax.set_xticks(ofs_steps)
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "answerability_factscore_overall.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "png" / "answerability_factscore_overall.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: answerability_factscore_overall")

    print(f"\nAll plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
