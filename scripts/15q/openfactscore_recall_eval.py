"""
openfactscore_recall_eval.py — OFS recall variant.

Direction: facts are extracted from E_0 (original) and verified against
each E_t (t=1,2,3). This measures how many of the original facts
SURVIVE the rewriting, i.e. recall of source facts.

    Recall = |F_S ∩ F_G| / |F_S|

Complements openfactscore_eval.py which measures precision (facts in
the rewrite that are faithful to E_0).

Output:
  <input_stem>_openfactscore_recall.csv         — one row per (chain, step)
  <input_stem>_openfactscore_recall_details.csv — one row per (chain, step, fact)

Usage:
  python scripts/15q/openfactscore_recall_eval.py \
      --input results/300q/rewriting_chains_300q.csv \
      --use-4bit
"""

import argparse
import json
import re
import string
import time
from pathlib import Path

import nltk
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV   = REPO_ROOT / "results" / "15q" / "rewriting_chains_15q.csv"
DEFAULT_DEMOS = REPO_ROOT / "data" / "demons.json"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

AFG_MODEL_ID = "allenai/OLMo-2-1124-7B-SFT"
AFV_MODEL_ID = "google/gemma-3-4b-it"

K_BM25             = 1
AFG_MAX_NEW_TOKENS = 256
AFV_MAX_NEW_TOKENS = 8

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Sentence splitting (verbatim from openfactscore_eval.py)
# ---------------------------------------------------------------------------

def detect_initials(text):
    return [m.group() for m in re.finditer(r"[A-Z]\. ?[A-Z]\.", text)]


def fix_sentence_splitter(sentences, initials):
    for initial in initials:
        if not any(initial in s for s in sentences):
            alpha1, alpha2 = [s.strip() for s in initial.split(".") if s.strip()]
            for i, sent in enumerate(sentences):
                if sent.endswith(alpha1 + "."):
                    if i + 1 < len(sentences) and sentences[i + 1].startswith(alpha2 + "."):
                        sentences[i] = sent + " " + sentences[i + 1]
                        sentences = sentences[:i + 1] + sentences[i + 2:]
                        break
    merged = []
    for sent in sentences:
        if merged and len(sent.split()) <= 1:
            merged[-1] = merged[-1] + " " + sent
        else:
            merged.append(sent)
    return merged


def sentences_from_text(text):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences  = []
    for para in paragraphs:
        initials = detect_initials(para)
        curr     = sent_tokenize(para)
        curr     = fix_sentence_splitter(curr, initials)
        sentences.extend(curr)
    return sentences


# ---------------------------------------------------------------------------
# AFG / AFV prompts (verbatim from openfactscore_eval.py)
# ---------------------------------------------------------------------------

AFG_SYSTEM_INSTRUCT = (
    "\n                You are an annotator that breaks down sentences into "
    "independent facts, short statements that each contain one piece of "
    "information contained in the given sentence.\n"
    "                in the next paragraphs you have examples of sentences "
    "broken down in atomic facts. \n"
    "                You have to complete the example given by the user.\n"
    "                Do not add new entities, do not deviate from the subject "
    "of the sentence given by the user, do not hallucinate, do not repeat "
    "facts in the system prompt.\n"
    "                List the sentences using -\n                "
)

AFV_SYSTEM_INSTRUCT = (
    "You are an annotator that verifies the factuality of a sentence "
    "according to a given source text. You answer only True or False and "
    "provides no further explanations."
)


def build_afg_demos_block(demons, demon_keys, bm25, sentence):
    top_matches = bm25.get_top_n(sentence.split(" "), demon_keys, K_BM25)
    parts = []
    for match in top_matches:
        parts.append(f"Please breakdown the following sentence into independent facts: {match}")
        for fact in demons[match]:
            parts.append(f"- {fact}")
        parts.append("")
    return "\n".join(parts).rstrip("\n")


def parse_atomic_facts(generated_text):
    text  = generated_text.replace("<|eot_id|>", "")
    text  = re.sub(r"-\s*\n", "", text)
    facts = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            fact = line[2:].strip()
            if fact and fact[-1] not in string.punctuation:
                fact += "."
            if fact:
                facts.append(fact)
        elif not line:
            if facts:
                break
        else:
            if facts:
                break
    return facts


def build_afv_user_prompt(topic, source, claim):
    definition  = f"Answer the question about {topic} based on the given context.\n\n"
    definition += f"Title: {topic}\nText: {source.strip()}\n\n".strip()
    if definition[-1] not in string.punctuation:
        definition += "."
    return f"{definition}\n\nInput: {claim.strip()} True or False?\nAnswer:"


def parse_afv_label(generated_text):
    answer = generated_text.lower()
    if "true" in answer or "false" in answer:
        if "true" in answer and "false" not in answer:
            return "SURVIVED"
        elif "false" in answer and "true" not in answer:
            return "LOST"
        else:
            return "SURVIVED" if answer.index("true") > answer.index("false") else "LOST"
    stripped = answer.translate(str.maketrans("", "", string.punctuation)).split()
    return "LOST" if any(kw in stripped for kw in ("not", "cannot", "unknown", "information")) else "SURVIVED"


# ---------------------------------------------------------------------------
# Model wrapper (verbatim from openfactscore_eval.py)
# ---------------------------------------------------------------------------

class HFChatModel:
    def __init__(self, model_id, role_label, use_4bit: bool = False):
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
        print(
            f"[{role_label}] loaded in {time.time()-t0:.1f}s · device map: "
            f"{getattr(self.model, 'hf_device_map', 'n/a')}",
            flush=True,
        )

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
            text = (f"{system_prompt}\n\n" if system_prompt else "") + user_prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out    = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Core: extract E_0 facts once, verify against each E_t
# ---------------------------------------------------------------------------

def extract_atomic_facts(afg_model, text, demons, demon_keys, bm25):
    facts = []
    for sent in sentences_from_text(text):
        demos_block   = build_afg_demos_block(demons, demon_keys, bm25, sent)
        system_prompt = f"{AFG_SYSTEM_INSTRUCT}\n{demos_block}"
        user_prompt   = f"Please breakdown the following sentence into independent facts: {sent}"
        out           = afg_model.generate(system_prompt, user_prompt, AFG_MAX_NEW_TOKENS)
        facts.extend(parse_atomic_facts(out))
    return facts


def compute_recall(afv_model, e0_facts, et_text, topic):
    """
    Verify each E_0 fact against E_t.
    Returns recall = n_survived / n_facts.
    """
    if not e0_facts:
        return {"n_facts": 0, "n_survived": 0, "n_lost": 0,
                "recall_score": None, "verified_facts": []}

    verified = []
    for fact in e0_facts:
        user_prompt = build_afv_user_prompt(topic, et_text, fact)
        out         = afv_model.generate(AFV_SYSTEM_INSTRUCT, user_prompt, AFV_MAX_NEW_TOKENS)
        label       = parse_afv_label(out)
        verified.append({"fact": fact, "label": label, "raw": out})

    n_survived = sum(1 for v in verified if v["label"] == "SURVIVED")
    return {
        "n_facts":        len(e0_facts),
        "n_survived":     n_survived,
        "n_lost":         len(e0_facts) - n_survived,
        "recall_score":   n_survived / len(e0_facts),
        "verified_facts": verified,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OFS recall: facts extracted from E_0, verified against each E_t."
    )
    parser.add_argument("--input",     type=Path, default=DEFAULT_CSV)
    parser.add_argument("--demos",     type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--afg-model", default=AFG_MODEL_ID)
    parser.add_argument("--afv-model", default=AFV_MODEL_ID)
    parser.add_argument("--limit",     type=int,  default=None)
    parser.add_argument("--qid",       action="append", default=None)
    parser.add_argument("--use-4bit",  action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Chain CSV not found: {args.input}")
    if not args.demos.exists():
        raise FileNotFoundError(f"demons.json not found: {args.demos}")

    print("=" * 70)
    print("OFS Recall — E_0 facts verified against E_t")
    print(f"  Recall = |F_S ∩ F_G| / |F_S|")
    print(f"  AFG: {args.afg_model}")
    print(f"  AFV: {args.afv_model}")
    print("=" * 70)

    with open(args.demos) as f:
        demons = json.load(f)
    demon_keys = list(demons.keys())
    bm25       = BM25Okapi([doc.split(" ") for doc in demon_keys])
    print(f"\nLoaded {len(demon_keys)} AFG demonstrations")

    df      = pd.read_csv(args.input)
    df      = df.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)
    sources = df[df["step"] == 0].set_index(CHAIN_KEYS)["text"].to_dict()
    print(f"Loaded {len(df)} rows · {len(sources)} chains")

    to_eval = df[df["step"] > 0].copy()
    if args.qid:
        to_eval = to_eval[to_eval["qid"].isin(args.qid)]
    if args.limit:
        to_eval = to_eval.head(args.limit)
        print(f"*** SMOKE TEST: limiting to first {args.limit} rows ***")

    out_scores  = args.input.with_name(args.input.stem + "_openfactscore_recall.csv")
    out_details = args.input.with_name(args.input.stem + "_openfactscore_recall_details.csv")
    print(f"Output scores:  {out_scores}")
    print(f"Output details: {out_details}")

    done_keys = set()
    if out_scores.exists():
        prev      = pd.read_csv(out_scores)
        done_keys = {tuple(row[k] for k in CHAIN_KEYS + ["step"]) for _, row in prev.iterrows()}
        print(f"Resume: {len(done_keys)} rows already scored.")

    afg = HFChatModel(args.afg_model, "AFG", use_4bit=args.use_4bit)
    afv = HFChatModel(args.afv_model, "AFV", use_4bit=args.use_4bit)

    # E_0 facts are the same for all instructions/runs of a given qid —
    # extract once per qid and cache.
    e0_facts_cache: dict = {}

    total   = len(to_eval)
    t_start = time.time()
    n_done  = 0

    for i, (_, row) in enumerate(to_eval.iterrows(), start=1):
        chain_id = tuple(row[k] for k in CHAIN_KEYS)
        key      = chain_id + (int(row["step"]),)
        if key in done_keys:
            continue

        e0_text = sources.get(chain_id)
        if e0_text is None:
            continue

        # Extract E_0 facts once per qid (shared across instructions/runs)
        qid = row["qid"]
        if qid not in e0_facts_cache:
            print(f"  [AFG] extracting E_0 facts for {qid} ...", flush=True)
            e0_facts_cache[qid] = extract_atomic_facts(afg, e0_text, demons, demon_keys, bm25)
            print(f"  [AFG] {len(e0_facts_cache[qid])} facts extracted", flush=True)

        label = f"{row['group']}/{row['instruction_type']}/run{row['run']}/step{row['step']}"
        t0    = time.time()
        print(f"[{i}/{total}] {label} ...", end=" ", flush=True)

        result  = compute_recall(afv, e0_facts_cache[qid], row["text"], qid)
        elapsed = time.time() - t0

        score_row = pd.DataFrame([{
            **{k: row[k] for k in CHAIN_KEYS},
            "step":         int(row["step"]),
            "n_facts":      result["n_facts"],
            "n_survived":   result["n_survived"],
            "n_lost":       result["n_lost"],
            "recall_score": result["recall_score"],
        }])
        score_row.to_csv(out_scores, mode="a", header=not out_scores.exists(), index=False)

        if result["verified_facts"]:
            details_rows = pd.DataFrame([
                {**{k: row[k] for k in CHAIN_KEYS},
                 "step": int(row["step"]),
                 "fact": vf["fact"], "label": vf["label"], "raw": vf["raw"]}
                for vf in result["verified_facts"]
            ])
            details_rows.to_csv(out_details, mode="a", header=not out_details.exists(), index=False)

        n_done += 1
        if result["recall_score"] is None:
            print(f"no facts  [{elapsed:.1f}s]")
        else:
            print(
                f"facts={result['n_facts']:>2}  survived={result['n_survived']:>2}  "
                f"lost={result['n_lost']:>2}  recall={result['recall_score']:.3f}  "
                f"[{elapsed:.1f}s]",
                flush=True,
            )

        avg       = (time.time() - t_start) / max(n_done, 1)
        remaining = (total - i) * avg
        if i % 10 == 0:
            print(f"   ETA: {remaining/60:.1f} min  (avg {avg:.1f}s/row)", flush=True)

    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")
    print(f"Saved: {out_scores}")
    print(f"Saved: {out_details}")

    print("\n" + "=" * 70)
    print("OFS Recall — median per (instruction_type, step)")
    print("=" * 70)
    out   = pd.read_csv(out_scores)
    pivot = out.pivot_table(
        index="instruction_type", columns="step", values="recall_score", aggfunc="median",
    )
    print(pivot.round(3))


if __name__ == "__main__":
    main()
