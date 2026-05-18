"""Plot OpenFactScore across rewriting steps — by instruction_type and by n_hop."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "results/300q/rewriting_chains_300q_openfactscore.csv"
OUT_INSTR = REPO / "results/300q/ofs_by_instruction_300q.png"
OUT_HOP   = REPO / "results/300q/ofs_by_hop_300q.png"

INSTR_ORDER = ["formality", "paraphrase", "shorten", "elaborate"]
INSTR_COLORS = {
    "formality": "#1f77b4",
    "paraphrase": "#2ca02c",
    "shorten":    "#d62728",
    "elaborate":  "#9467bd",
}

HOP_ORDER  = [2, 3, 4]
HOP_COLORS = {2: "#1f77b4", 3: "#ff7f0e", 4: "#2ca02c"}

df = pd.read_csv(CSV)
df["n_hop"] = df["qid"].str.extract(r"^(\d+)hop").astype(int)


def agg_by(df, groupcol):
    agg = (
        df.groupby([groupcol, "step"])["factscore"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / agg["count"].pow(0.5)
    return agg


def make_plot(agg, groupcol, order, colors, title, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key in order:
        sub = agg[agg[groupcol] == key].sort_values("step")
        if sub.empty:
            continue
        ax.errorbar(
            sub["step"],
            sub["mean"],
            yerr=sub["sem"],
            marker="o",
            linewidth=2,
            capsize=3,
            label=str(key),
            color=colors[key],
        )
    ax.set_xlabel("Rewriting step")
    ax.set_ylabel("OpenFactScore (mean ± SEM)")
    ax.set_title(title)
    ax.set_xticks(sorted(df["step"].unique()))
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(title=groupcol, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")


make_plot(
    agg_by(df, "instruction_type"),
    "instruction_type",
    INSTR_ORDER,
    INSTR_COLORS,
    "OpenFactScore across rewriting steps — 300q (by instruction type)",
    OUT_INSTR,
)

make_plot(
    agg_by(df, "n_hop"),
    "n_hop",
    HOP_ORDER,
    HOP_COLORS,
    "OpenFactScore across rewriting steps — 300q (by n_hop)",
    OUT_HOP,
)
