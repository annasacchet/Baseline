"""
OpenFActScore — RECALL for 600q.

Reuses the atomic facts already extracted from each E_0 source text by the
forward OpenFActScore pipeline (rows with label='E0' in the
rewriting_chains_musique_600q_openfactscore_details.csv file). For each
rewritten step E_k (k in {1,2,3}) and each E_0 fact, runs AFV to check whether
the fact is entailed by E_k.

recall_init = n_recalled / n_e0_facts
recall      = recall_init * min(1, n_e0_facts / gamma)   [length penalty]

Output:
  results/600q/rewriting_chains_musique_600q_openfactscore_recall.csv
    qid, group, instruction_type, run, step, instruction_used,
    n_e0_facts, n_recalled, n_not_recalled, recall_init, recall

  results/600q/rewriting_chains_musique_600q_openfactscore_recall_details.csv
    qid, group, instruction_type, run, step, fact, label, raw
"""

import argparse
import string
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHAIN_CSV = REPO_ROOT / "results" / "600q" / "rewriting_chains_musique_600q.csv"
DEFAULT_DETAILS_CSV = (
    REPO_ROOT / "results" / "600q" / "rewriting_chains_musique_600q_openfactscore_details.csv"
)

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

AFV_MODEL_ID = "google/gemma-3-4b-it"
AFV_MAX_NEW_TOKENS = 8


AFV_SYSTEM_INSTRUCT = (
    "You are an annotator that verifies the factuality of a sentence "
    "according to a given source text. You answer only True or False and "
    "provides no further explanations."
)


def build_afv_user_prompt(topic, source, claim):
    definition = f"Answer the question about {topic} based on the given context.\n\n"
    context = f"Title: {topic}\nText: {source.strip()}\n\n"
    definition += context.strip()
    if definition[-1] not in string.punctuation:
        definition += "."
    return f"{definition.strip()}\n\nInput: {claim.strip()} True or False?\nAnswer:"


def parse_afv_label(generated_text):
    answer = generated_text.lower()
    if "true" in answer or "false" in answer:
        if "true" in answer and "false" not in answer:
            is_supported = True
        elif "false" in answer and "true" not in answer:
            is_supported = False
        else:
            is_supported = answer.index("true") > answer.index("false")
    else:
        stripped = answer.translate(str.maketrans("", "", string.punctuation)).split()
        is_supported = all(
            kw not in stripped for kw in ("not", "cannot", "unknown", "information")
        )
    return "SUPPORTED" if is_supported else "NOT_SUPPORTED"


class HFChatModel:
    def __init__(self, model_id, role_label, use_4bit=False):
        print(f"[{role_label}] loading {model_id} (4-bit={use_4bit}) ...", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"device_map": "auto", "trust_remote_code": True}
        if use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        print(f"[{role_label}] loaded in {time.time()-t0:.1f}s", flush=True)

    @torch.no_grad()
    def generate(self, system_prompt, user_prompt, max_new_tokens):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            sys_block = f"{system_prompt}\n\n" if system_prompt else ""
            text = f"{sys_block}{user_prompt}"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compute_recall(afv, e0_facts, ek_text, topic, gamma=10):
    if not e0_facts:
        return {"n_e0_facts": 0, "n_recalled": 0, "n_not_recalled": 0,
                "recall_init": None, "recall": None, "verified_facts": []}
    verified = []
    for fact in e0_facts:
        user_prompt = build_afv_user_prompt(topic, ek_text, fact)
        out = afv.generate(AFV_SYSTEM_INSTRUCT, user_prompt, AFV_MAX_NEW_TOKENS)
        verified.append({"fact": fact, "label": parse_afv_label(out), "raw": out})
    n_recalled = sum(1 for v in verified if v["label"] == "SUPPORTED")
    recall_init = n_recalled / len(e0_facts)
    length_penalty = min(1.0, len(e0_facts) / gamma) if gamma > 0 else 1.0
    return {
        "n_e0_facts": len(e0_facts),
        "n_recalled": n_recalled,
        "n_not_recalled": len(e0_facts) - n_recalled,
        "recall_init": recall_init,
        "recall": recall_init * length_penalty,
        "verified_facts": verified,
    }


def load_e0_facts_from_details(details_csv):
    """Read the E0-labelled rows from the forward OFS details CSV and return
    {qid: [fact, ...]}. Facts are de-duplicated within each qid preserving
    insertion order."""
    df = pd.read_csv(details_csv)
    e0 = df[df["label"] == "E0"]
    if e0.empty:
        raise ValueError(f"No rows with label='E0' in {details_csv}")
    cache = {}
    for qid, grp in e0.groupby("qid", sort=False):
        seen = set()
        ordered = []
        for fact in grp["fact"].tolist():
            if fact not in seen:
                seen.add(fact)
                ordered.append(fact)
        cache[qid] = ordered
    return cache


def main():
    parser = argparse.ArgumentParser(
        description="OFS Recall for 600q — reuses E_0 facts from the forward OFS details CSV."
    )
    parser.add_argument("--chain-csv", type=Path, default=DEFAULT_CHAIN_CSV,
                        help="Rewriting chains CSV (must contain step>0 rows with text).")
    parser.add_argument("--details-csv", type=Path, default=DEFAULT_DETAILS_CSV,
                        help="Forward OFS details CSV (source of E_0 facts).")
    parser.add_argument("--afv-model", default=AFV_MODEL_ID)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke-test: process only first N qids.")
    parser.add_argument("--qid", action="append", default=None,
                        help="Process only these qids (can be repeated).")
    parser.add_argument("--gamma", type=int, default=10,
                        help="Length-penalty gamma (default 10).")
    args = parser.parse_args()

    if not args.chain_csv.exists():
        raise FileNotFoundError(f"Chain CSV not found: {args.chain_csv}")
    if not args.details_csv.exists():
        raise FileNotFoundError(f"Details CSV not found: {args.details_csv}")

    print("=" * 70)
    print("OpenFActScore RECALL (600q) — AFV-only, reusing E_0 facts")
    print(f"  AFV: {args.afv_model}  4-bit={args.use_4bit}  gamma={args.gamma}")
    print("=" * 70)

    e0_facts_cache = load_e0_facts_from_details(args.details_csv)
    print(f"Loaded E_0 facts for {len(e0_facts_cache)} qids "
          f"(avg {sum(len(v) for v in e0_facts_cache.values())/max(len(e0_facts_cache),1):.1f} facts/qid)")

    chain = pd.read_csv(args.chain_csv)
    chain = chain.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    to_eval = chain[chain["step"] > 0].copy()
    # Only keep chains whose qid has E_0 facts available.
    to_eval = to_eval[to_eval["qid"].isin(e0_facts_cache.keys())]

    if args.qid:
        to_eval = to_eval[to_eval["qid"].isin(args.qid)]
        e0_facts_cache = {k: v for k, v in e0_facts_cache.items() if k in args.qid}
    if args.limit:
        qids_limited = list(e0_facts_cache.keys())[:args.limit]
        e0_facts_cache = {k: v for k, v in e0_facts_cache.items() if k in qids_limited}
        to_eval = to_eval[to_eval["qid"].isin(qids_limited)]
        print(f"*** SMOKE TEST: first {args.limit} qids → {len(to_eval)} rows ***")

    out_scores = args.chain_csv.with_name(
        args.chain_csv.stem + "_openfactscore_recall.csv"
    )
    out_details = args.chain_csv.with_name(
        args.chain_csv.stem + "_openfactscore_recall_details.csv"
    )
    print(f"Recall scores:  {out_scores}")
    print(f"Recall details: {out_details}")

    done_keys = set()
    if out_scores.exists():
        prev_scores = pd.read_csv(out_scores)
        done_keys = {tuple(r[k] for k in CHAIN_KEYS + ["step"]) for _, r in prev_scores.iterrows()}
        print(f"Resume: {len(done_keys)} (chain, step) already scored.")

    afv = HFChatModel(args.afv_model, "AFV", use_4bit=args.use_4bit)

    total = len(to_eval)
    print(f"\n--- AFV on {total} rewritten steps ---")
    t0_all = time.time()
    n_done = 0

    for i, (_, row) in enumerate(to_eval.iterrows(), start=1):
        chain_id = tuple(row[k] for k in CHAIN_KEYS)
        key = chain_id + (int(row["step"]),)
        if key in done_keys:
            continue

        e0_facts = e0_facts_cache.get(row["qid"], [])
        label_str = (
            f"{row['group']}/{row['instruction_type']}/run{int(row['run'])}/step{int(row['step'])}"
        )
        print(f"[{i}/{total}] {row['qid']}  {label_str}  ({len(e0_facts)} facts) ...",
              end=" ", flush=True)
        t0 = time.time()

        result = compute_recall(afv, e0_facts, row["text"], row["qid"], gamma=args.gamma)
        elapsed = time.time() - t0

        score_row = pd.DataFrame([{
            **{k: row[k] for k in CHAIN_KEYS},
            "step": int(row["step"]),
            "instruction_used": row.get("instruction_used"),
            **{k: v for k, v in result.items() if k != "verified_facts"},
        }])
        score_row.to_csv(out_scores, mode="a", header=not out_scores.exists(),
                         index=False, encoding="utf-8")

        if result["verified_facts"]:
            pd.DataFrame([
                {**{k: row[k] for k in CHAIN_KEYS},
                 "step": int(row["step"]), "fact": vf["fact"],
                 "label": vf["label"], "raw": vf["raw"]}
                for vf in result["verified_facts"]
            ]).to_csv(out_details, mode="a", header=not out_details.exists(),
                      index=False, encoding="utf-8")

        n_done += 1
        if result["recall_init"] is None:
            print(f"no facts [{elapsed:.1f}s]")
        else:
            print(
                f"recalled={result['n_recalled']:>3}/{result['n_e0_facts']:>3}  "
                f"recall_init={result['recall_init']:.3f}  [{elapsed:.1f}s]",
                flush=True,
            )

        if i % 10 == 0:
            avg = (time.time() - t0_all) / max(n_done, 1)
            print(f"   ETA: {(total - i) * avg / 60:.1f} min  (avg {avg:.1f}s/row)",
                  flush=True)

    print(f"\nDone in {(time.time()-t0_all)/60:.1f} min")
    print(f"Saved: {out_scores}")
    print(f"Saved: {out_details}")

    if out_scores.exists():
        print("\n" + "=" * 70)
        print("OFS Recall — median recall_init per (instruction_type, step)")
        print("=" * 70)
        out = pd.read_csv(out_scores)
        pivot = out.pivot_table(
            index="instruction_type", columns="step", values="recall_init", aggfunc="median",
        )
        print(pivot.round(3))


if __name__ == "__main__":
    main()
