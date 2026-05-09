"""
End-to-end smoke test of the rewriting pipeline using OpenAI gpt-4o-mini.

Goal
----
Run the FULL pipeline (rewriting → answer F1 → BERTScore → OpenFActScore + length)
on a tiny sample (default: 2 stories × 1 wording per instruction = 4 chains)
to verify that the pipeline code works end-to-end. Compares both datasets
(NewsQA and MuSiQue) using the same model so any divergence is dataset-side.

Why this exists
---------------
Before burning hours of GPU time on the 32B OLMo runs, we want to know:
  - does each pipeline component actually produce the expected output schema?
  - do F1 / OpenFActScore numbers move in plausible directions across steps?
  - are there NewsQA/MuSiQue-specific bugs (e.g. answer parsing, alias merging)?

If the OpenAI runs look sane on both datasets, the GPU run can be trusted.
If they look broken on NewsQA but fine on MuSiQue → bug in the NewsQA loader.

Usage
-----
  OPENAI_API_KEY=sk-... python scripts/smoke_openai/smoke_test_pipeline.py --dataset newsqa
  OPENAI_API_KEY=sk-... python scripts/smoke_openai/smoke_test_pipeline.py --dataset musique

Outputs go to results/smoke_openai/<dataset>/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse existing pipeline pieces (no duplication of business logic).
from openai_chat import OpenAIChat  # noqa: E402

# Reuse the OpenFActScore primitives from the 15q script (same module API).
sys.path.insert(0, str(REPO_ROOT / "scripts" / "15q"))
from openfactscore_eval import (  # noqa: E402
    AFG_SYSTEM_INSTRUCT, AFV_SYSTEM_INSTRUCT,
    AFG_MAX_NEW_TOKENS, AFV_MAX_NEW_TOKENS,
    K_BM25, build_afg_demos_block, build_afv_user_prompt,
    parse_atomic_facts, parse_afv_label, sentences_from_text,
)

# Reuse NewsQA loader for dataset selection.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "newsqa"))
from rewriting_pipeline_newsqa import (  # noqa: E402
    ALL_INSTRUCTIONS, REWRITE_TEMPLATE, ALIAS_SEP,
    load_newsqa, sample_items as newsqa_sample_items,
)

# Reuse FictionalQA loader.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "fictionalqa"))
from rewriting_pipeline_fictionalqa import (  # noqa: E402
    load_fictionalqa, sample_items as fictionalqa_sample_items,
)

from rank_bm25 import BM25Okapi  # noqa: E402


CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]
DEFAULT_DEMOS = REPO_ROOT / "data" / "demons.json"
DEFAULT_NEWSQA_CSV = Path("/Users/annasacchet/combined-newsqa-data-v1.csv")
DEFAULT_MUSIQUE_JSONL = Path("/Users/annasacchet/Desktop/musique_ans_v1.0_dev.jsonl")


# ---------------------------------------------------------------------------
# Dataset adapters — return a list of records: {qid, question, gold_answer,
# gold_aliases, E0, topic} so the rest of the pipeline is dataset-agnostic.
# ---------------------------------------------------------------------------

def load_newsqa_sample(csv_path: Path, n: int, seed: int) -> list[dict]:
    items = load_newsqa(csv_path)
    sampled = newsqa_sample_items(items, n, seed)
    out = []
    for it in sampled:
        # NewsQA storyId paths make poor topics; use the article's first line.
        topic = next((ln.strip() for ln in it["text"].splitlines() if ln.strip()),
                     it["id"])[:200]
        out.append({
            "qid": it["id"],
            "question": it["question"],
            "gold_answer": it["answer"],
            "gold_aliases": it["aliases"],
            "E0": it["text"],
            "topic": topic,
        })
    return out


def load_fictionalqa_sample(n: int, seed: int) -> list[dict]:
    """FictionalQA sample — pulls fictions + joined_qa from HF, picks best Q per doc."""
    items = load_fictionalqa()
    sampled = fictionalqa_sample_items(items, n, seed)
    out = []
    for it in sampled:
        topic = next((ln.strip() for ln in it["text"].splitlines() if ln.strip()),
                     it["id"])[:200]
        out.append({
            "qid": it["id"],
            "question": it["question"],
            "gold_answer": it["answer"],
            "gold_aliases": it["aliases"],
            "E0": it["text"],
            "topic": topic,
        })
    return out


def load_musique_sample(jsonl_path: Path, n: int, seed: int) -> list[dict]:
    items = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(items)
    out = []
    for it in items[:n]:
        # Concatenate ALL paragraphs (including distractors) — matches the
        # production pipeline default in scripts/15q/rewriting_pipeline.py.
        E0 = "\n\n".join(
            f"{p['title']}. {p['paragraph_text']}" for p in it["paragraphs"]
        )
        out.append({
            "qid": it["id"],
            "question": it["question"],
            "gold_answer": it["answer"],
            "gold_aliases": [it["answer"]] + (it.get("answer_aliases") or []),
            "E0": E0,
            "topic": it["question"],  # MuSiQue questions are good topic strings
        })
    return out


# ---------------------------------------------------------------------------
# Stage 1 — rewriting chains
# ---------------------------------------------------------------------------

def run_rewriting(
    records: list[dict],
    rewriter: OpenAIChat,
    n_iterations: int,
    instructions: dict,
    out_csv: Path,
    max_new_tokens: int,
) -> pd.DataFrame:
    rows = []
    total = len(records) * sum(len(p) for p in instructions.values())
    n_done = 0
    t_start = time.time()
    for r in records:
        for (group, instruction_type), pool in instructions.items():
            for run, instruction in enumerate(pool):
                chain = [r["E0"]]
                current = r["E0"]
                for _ in range(n_iterations):
                    prompt = REWRITE_TEMPLATE.format(instruction=instruction, text=current)
                    current = rewriter.complete(prompt, max_tokens=max_new_tokens)
                    chain.append(current)
                for step, text in enumerate(chain):
                    rows.append({
                        "qid": r["qid"],
                        "question": r["question"],
                        "gold_answer": r["gold_answer"],
                        "gold_answer_aliases": ALIAS_SEP.join(r["gold_aliases"]),
                        "group": group,
                        "instruction_type": instruction_type,
                        "run": run,
                        "instruction_used": instruction if step > 0 else "",
                        "step": step,
                        "text": text,
                        # No tokenizer here; use whitespace count as a coarse proxy.
                        "n_tokens": len(text.split()),
                    })
                n_done += 1
                avg = (time.time() - t_start) / n_done
                print(f"  [{n_done}/{total}] {r['qid'][-25:]} | {group}/{instruction_type}/run{run}"
                      f"  avg {avg:.1f}s/chain", flush=True)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"  saved chains: {out_csv}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Stage 2 — Answer F1 (uses the same SQuAD-style normalization as both pipelines)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    import re, string
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(gold: str, pred: str) -> float:
    g = normalize_answer(gold).split()
    p = normalize_answer(pred).split()
    if not g or not p:
        return float(g == p)
    common = Counter(g) & Counter(p)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


QA_TEMPLATE = """Answer the question based on the context below. Give a short, direct answer — a few words at most, no explanation.

Context:
{context}

Question: {question}
Answer:"""


def run_answer_f1(chains_df: pd.DataFrame, qa: OpenAIChat, out_csv: Path) -> pd.DataFrame:
    results = []
    total = len(chains_df)
    for i, row in enumerate(chains_df.itertuples(index=False), start=1):
        prompt = QA_TEMPLATE.format(context=row.text.strip(), question=row.question.strip())
        pred = qa.complete(prompt, max_tokens=96)
        aliases = [a for a in str(row.gold_answer_aliases).split(ALIAS_SEP) if a]
        if row.gold_answer not in aliases:
            aliases = [row.gold_answer] + aliases
        best = max((compute_f1(g, pred) for g in aliases), default=0.0)
        matched = max(aliases, key=lambda g: compute_f1(g, pred)) if aliases else ""
        results.append({
            "qid": row.qid, "group": row.group, "instruction_type": row.instruction_type,
            "run": int(row.run), "step": int(row.step),
            "question": row.question, "gold_answer": row.gold_answer,
            "predicted_answer": pred, "matched_reference": matched, "answer_f1": best,
        })
        print(f"  [{i}/{total}] step={row.step} F1={best:.3f}  pred={pred[:40]!r}", flush=True)
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"  saved F1: {out_csv}")
    return df


# ---------------------------------------------------------------------------
# Stage 3 — BERTScore (always uses the local roberta-large encoder)
# ---------------------------------------------------------------------------

def run_bertscore(chains_df: pd.DataFrame, out_csv: Path, device: str | None = None) -> pd.DataFrame:
    import torch
    from bert_score import score as compute_bert_score

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"  BERTScore device: {device}")

    # Build lookup
    idx = {(r.qid, r.group, r.instruction_type, int(r.run), int(r.step)): r.text
           for r in chains_df.itertuples(index=False)}

    cands_b, refs_b, cands_c, refs_c, keys = [], [], [], [], []
    for r in chains_df.itertuples(index=False):
        if int(r.step) == 0:
            continue
        key = (r.qid, r.group, r.instruction_type, int(r.run), int(r.step))
        e0 = idx.get((r.qid, r.group, r.instruction_type, int(r.run), 0))
        eprev = idx.get((r.qid, r.group, r.instruction_type, int(r.run), int(r.step) - 1))
        if e0 is None or eprev is None:
            continue
        cands_b.append(r.text); refs_b.append(e0)
        cands_c.append(r.text); refs_c.append(eprev)
        keys.append(key)

    Pb, Rb, Fb = compute_bert_score(cands_b, refs_b, lang="en",
                                    model_type="roberta-large", num_layers=17,
                                    batch_size=4, device=device, verbose=False)
    Pc, Rc, Fc = compute_bert_score(cands_c, refs_c, lang="en",
                                    model_type="roberta-large", num_layers=17,
                                    batch_size=4, device=device, verbose=False)

    rows = []
    for k, pb, rb, fb, pc, rc, fc in zip(
        keys, Pb.tolist(), Rb.tolist(), Fb.tolist(),
              Pc.tolist(), Rc.tolist(), Fc.tolist()):
        qid, g, it, run, step = k
        rows.append({
            "qid": qid, "group": g, "instruction_type": it, "run": run, "step": step,
            "bert_precision_baseline": pb, "bert_recall_baseline": rb, "bert_f1_baseline": fb,
            "bert_precision_consecutive": pc, "bert_recall_consecutive": rc, "bert_f1_consecutive": fc,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  saved BERTScore: {out_csv}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Stage 4 — OpenFActScore with gpt-4o-mini as both AFG and AFV
# ---------------------------------------------------------------------------

def extract_facts(afg: OpenAIChat, text: str, demons, demon_keys, bm25) -> list[str]:
    sentences = sentences_from_text(text)
    all_facts = []
    for sent in sentences:
        demos_block = build_afg_demos_block(demons, demon_keys, bm25, sent)
        system = f"{AFG_SYSTEM_INSTRUCT}\n{demos_block}"
        user = f"Please breakdown the following sentence into independent facts: {sent}"
        out = afg.generate(system, user, AFG_MAX_NEW_TOKENS)
        all_facts.extend(parse_atomic_facts(out))
    return all_facts


def verify_facts(afv: OpenAIChat, source: str, facts: list[str], topic: str) -> list[dict]:
    results = []
    for fact in facts:
        prompt = build_afv_user_prompt(topic, source, fact)
        out = afv.generate(AFV_SYSTEM_INSTRUCT, prompt, AFV_MAX_NEW_TOKENS)
        results.append({"fact": fact, "label": parse_afv_label(out), "raw": out})
    return results


OFS_SCHEMA = [
    "qid", "group", "instruction_type", "run", "step",
    "n_facts_et", "n_supported_et", "n_not_supported_et", "init_score",
    "n_facts_e0", "n_supported_e0_in_et", "recall_score",
]


def _append_ofs_row(out_csv: Path, row: dict) -> None:
    """Append a single OFS row to disk (with header on first write).

    Writing per-row means an interrupted run still leaves a partial CSV that
    can be picked back up via the resume logic in run_openfactscore.
    """
    df = pd.DataFrame([{k: row.get(k) for k in OFS_SCHEMA}])
    df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)


def run_openfactscore(
    chains_df: pd.DataFrame, records: list[dict], afg: OpenAIChat, afv: OpenAIChat,
    demos_path: Path, out_csv: Path,
) -> pd.DataFrame:
    """Compute precision (init_score) AND recall per (chain, step), with resume.

    - Precision (init_score): facts extracted from E_t that are supported by E_0.
      Penalises hallucinations introduced by the rewriter.
    - Recall: facts extracted from E_0 that are still supported by E_t.
      Penalises information loss across rewriting steps.

    E_0 facts are extracted ONCE per qid (they don't depend on the chain) and
    cached, so the recall pass only adds AFV calls (cheap), not AFG.

    Resume policy
    -------------
    Rows are appended to out_csv as soon as they're computed. On startup we
    read out_csv (if any) and skip (qid, group, instruction_type, run, step)
    keys that are already there. This means an interrupted run can be picked
    back up by simply re-running the same command.
    """
    with open(demos_path) as f:
        demons = json.load(f)
    demon_keys = list(demons.keys())
    bm25 = BM25Okapi([d.split(" ") for d in demon_keys])

    sources = chains_df[chains_df["step"] == 0].set_index(CHAIN_KEYS)["text"].to_dict()
    topics = {r["qid"]: r["topic"] for r in records}

    # ---- Resume: load already-done (chain, step) keys from disk
    done_keys: set[tuple] = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done_keys = {
            (r["qid"], r["group"], r["instruction_type"], int(r["run"]), int(r["step"]))
            for _, r in prev.iterrows()
        }
        print(f"  resume: {len(done_keys)} OFS rows already in {out_csv} — will skip them.")

    # ---- Step 1: extract E_0 facts once per qid (used for recall)
    print("  extracting E_0 facts (once per story) ...")
    e0_facts_by_qid: dict[str, list[str]] = {}
    for qid in chains_df["qid"].unique():
        e0_text = chains_df[(chains_df["qid"] == qid) & (chains_df["step"] == 0)]["text"].iloc[0]
        facts = extract_facts(afg, e0_text, demons, demon_keys, bm25)
        e0_facts_by_qid[qid] = facts
        print(f"    qid={qid[-25:]} → {len(facts)} E_0 facts", flush=True)

    eval_df = chains_df[chains_df["step"] > 0]
    total = len(eval_df)
    for i, r in enumerate(eval_df.itertuples(index=False), start=1):
        key = (r.qid, r.group, r.instruction_type, int(r.run), int(r.step))
        if key in done_keys:
            print(f"  [{i}/{total}] step={r.step} {r.group}/{r.instruction_type}/run{r.run} — already done, skip", flush=True)
            continue

        chain_id = (r.qid, r.group, r.instruction_type, int(r.run))
        source = sources.get(chain_id)  # E_0
        topic = topics.get(r.qid, r.qid)

        # ---- Precision: facts(E_t) verified against E_0
        et_facts = extract_facts(afg, r.text, demons, demon_keys, bm25)
        if et_facts:
            verified_p = verify_facts(afv, source, et_facts, topic)
            n_supp_p = sum(1 for v in verified_p if v["label"] == "SUPPORTED")
            init_score = n_supp_p / len(et_facts)
        else:
            n_supp_p = 0
            init_score = None

        # ---- Recall: facts(E_0) verified against E_t
        e0_facts = e0_facts_by_qid.get(r.qid, [])
        if e0_facts:
            verified_r = verify_facts(afv, r.text, e0_facts, topic)
            n_supp_r = sum(1 for v in verified_r if v["label"] == "SUPPORTED")
            recall_score = n_supp_r / len(e0_facts)
        else:
            n_supp_r = 0
            recall_score = None

        row = {
            "qid": r.qid, "group": r.group, "instruction_type": r.instruction_type,
            "run": int(r.run), "step": int(r.step),
            "n_facts_et": len(et_facts), "n_supported_et": n_supp_p,
            "n_not_supported_et": len(et_facts) - n_supp_p,
            "init_score": init_score,
            "n_facts_e0": len(e0_facts), "n_supported_e0_in_et": n_supp_r,
            "recall_score": recall_score,
        }
        _append_ofs_row(out_csv, row)

        prec_str = f"{init_score:.3f}" if init_score is not None else "n/a "
        rec_str = f"{recall_score:.3f}" if recall_score is not None else "n/a "
        print(
            f"  [{i}/{total}] step={r.step} et_facts={len(et_facts):>2} prec={prec_str}"
            f"  e0_facts={len(e0_facts):>2} rec={rec_str}",
            flush=True,
        )

    print(f"  saved OpenFActScore (precision + recall): {out_csv}")
    return pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(chains_df, f1_df, bert_df, ofs_df) -> None:
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # length / tokens by step
    print("\n[length proxy] mean n_tokens by step:")
    print(chains_df.groupby("step")["n_tokens"].mean().round(1).to_string())

    print("\n[answer F1] mean by step:")
    print(f1_df.groupby("step")["answer_f1"].mean().round(3).to_string())

    print("\n[BERTScore F1 vs E_0] mean by step:")
    print(bert_df.groupby("step")["bert_f1_baseline"].mean().round(3).to_string())

    print("\n[OpenFActScore precision (init_score) — facts in E_t supported by E_0] mean by step:")
    print(ofs_df.groupby("step")["init_score"].mean().round(3).to_string())

    print("\n[OpenFActScore recall — facts in E_0 supported by E_t] mean by step:")
    print(ofs_df.groupby("step")["recall_score"].mean().round(3).to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Smoke test the rewriting pipeline using OpenAI gpt-4o-mini.")
    p.add_argument("--dataset", choices=["newsqa", "musique", "fictionalqa"], required=True)
    p.add_argument("--n-stories", type=int, default=2)
    # ONE wording per instruction type by default = 4 chains/story instead of 12.
    p.add_argument("--one-wording-per-instruction", action="store_true", default=True)
    p.add_argument("--n-iterations", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rewriter-model", default="gpt-4o-mini")
    p.add_argument("--qa-model", default="gpt-4o-mini")
    p.add_argument("--afg-model", default="gpt-4o-mini")
    p.add_argument("--afv-model", default="gpt-4o-mini")
    p.add_argument("--max-rewrite-tokens", type=int, default=4096)
    p.add_argument("--bert-device", default=None)
    p.add_argument("--newsqa-csv", type=Path, default=DEFAULT_NEWSQA_CSV)
    p.add_argument("--musique-jsonl", type=Path, default=DEFAULT_MUSIQUE_JSONL)
    p.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    p.add_argument("--skip-bertscore", action="store_true")
    p.add_argument("--skip-openfactscore", action="store_true")
    args = p.parse_args()

    out_dir = REPO_ROOT / "results" / "smoke_openai" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"SMOKE TEST · dataset={args.dataset} · {args.n_stories} stories · "
          f"{args.n_iterations} iterations")
    print(f"  rewriter:  {args.rewriter_model}")
    print(f"  QA:        {args.qa_model}")
    print(f"  AFG/AFV:   {args.afg_model} / {args.afv_model}")
    print(f"  output:    {out_dir}")
    print("=" * 70)

    # ---- load sample
    print("\n[1/5] loading dataset sample ...")
    if args.dataset == "newsqa":
        records = load_newsqa_sample(args.newsqa_csv, args.n_stories, args.seed)
    elif args.dataset == "fictionalqa":
        records = load_fictionalqa_sample(args.n_stories, args.seed)
    else:
        records = load_musique_sample(args.musique_jsonl, args.n_stories, args.seed)
    print(f"  got {len(records)} records")
    for r in records:
        print(f"    qid={r['qid'][-30:]}  Q={r['question'][:60]!r}  A={r['gold_answer'][:40]!r}")

    # restrict to one wording per (group, instruction_type)
    if args.one_wording_per_instruction:
        instructions = {k: v[:1] for k, v in ALL_INSTRUCTIONS.items()}
    else:
        instructions = ALL_INSTRUCTIONS

    # ---- rewriting
    print("\n[2/5] rewriting chains ...")
    rewriter = OpenAIChat(args.rewriter_model, role_label="rewriter")
    chains_df = run_rewriting(
        records, rewriter, args.n_iterations, instructions,
        out_dir / "chains.csv", args.max_rewrite_tokens,
    )

    # ---- answer F1
    print("\n[3/5] answer F1 ...")
    qa = OpenAIChat(args.qa_model, role_label="QA")
    f1_df = run_answer_f1(chains_df, qa, out_dir / "answer_f1.csv")

    # ---- BERTScore
    if args.skip_bertscore:
        print("\n[4/5] skipping BERTScore (per --skip-bertscore)")
        bert_df = pd.DataFrame()
    else:
        print("\n[4/5] BERTScore ...")
        bert_df = run_bertscore(chains_df, out_dir / "bertscore.csv", device=args.bert_device)

    # ---- OpenFActScore
    if args.skip_openfactscore:
        print("\n[5/5] skipping OpenFActScore (per --skip-openfactscore)")
        ofs_df = pd.DataFrame()
    else:
        print("\n[5/5] OpenFActScore ...")
        afg = OpenAIChat(args.afg_model, role_label="AFG")
        afv = OpenAIChat(args.afv_model, role_label="AFV")
        ofs_df = run_openfactscore(chains_df, records, afg, afv,
                                    args.demos, out_dir / "openfactscore.csv")

    if not bert_df.empty and not ofs_df.empty:
        print_summary(chains_df, f1_df, bert_df, ofs_df)


if __name__ == "__main__":
    main()
