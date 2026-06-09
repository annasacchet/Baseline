"""OpenFActScore (source-faithfulness) — OLMo-3.1-32B rewriting chains.

This script is dataset-agnostic: it reads any chain CSV and scores every
(step > 0) row against E_0 by extracting atomic facts (AFG) and verifying
each against the source (AFV).

What we keep vs the original 600q OFS configuration
---------------------------------------------------
OFS uses two models we keep pinned for consistency with prior runs:
  - AFG: allenai/OLMo-2-1124-7B-SFT  (same as 600q/newsqa pipelines)
  - AFV: google/gemma-3-4b-it        (kept for AFV-consistency, see
                                       feedback_afv_model_consistency)

If you want to swap AFG to OLMo-3.1-32B-Instruct too, pass
`--afg-model allenai/OLMo-3.1-32B-Instruct` (optionally with --use-4bit
for memory-constrained nodes).

Topic derivation
----------------
  --topic-mode qid           : use the qid string as topic (MuSiQue default)
  --topic-mode first-line    : first non-empty line of E_0, ≤200 chars
                               (recommended for NewsQA + FictionalQA)
  --topic-mode question      : use the question text as the AFV topic
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path

import nltk
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "qwen30b"))
from _common.olmo_constants import AFV_MODEL_ID, CHAIN_KEYS  # noqa: E402

DEFAULT_DEMOS = REPO_ROOT / "data" / "demons.json"

AFG_MODEL_ID = "allenai/OLMo-2-1124-7B-SFT"
K_BM25 = 1
AFG_MAX_NEW_TOKENS = 256
AFV_MAX_NEW_TOKENS = 8
TOPIC_MAX_CHARS = 200

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Sentence splitting
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
    sentences = []
    for para in paragraphs:
        initials = detect_initials(para)
        curr = sent_tokenize(para)
        curr = fix_sentence_splitter(curr, initials)
        sentences.extend(curr)
    return sentences


# ---------------------------------------------------------------------------
# AFG / AFV prompts
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
    top = bm25.get_top_n(sentence.split(" "), demon_keys, K_BM25)
    parts = []
    for match in top:
        parts.append(f"Please breakdown the following sentence into independent facts: {match}")
        for fact in demons[match]:
            parts.append(f"- {fact}")
        parts.append("")
    return "\n".join(parts).rstrip("\n")


def parse_atomic_facts(generated_text):
    text = generated_text.replace("<|eot_id|>", "")
    text = re.sub(r"-\s*\n", "", text)
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
        is_supported = all(kw not in stripped for kw in ("not", "cannot", "unknown", "information"))
    return "SUPPORTED" if is_supported else "NOT_SUPPORTED"


# ---------------------------------------------------------------------------
# HF chat-model wrapper
# ---------------------------------------------------------------------------

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
        print(f"[{role_label}] loaded in {time.time()-t0:.1f}s · "
              f"device map: {getattr(self.model, 'hf_device_map', 'n/a')}", flush=True)

    @torch.no_grad()
    def generate(self, system_prompt, user_prompt, max_new_tokens):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            sys_block = f"{system_prompt}\n\n" if system_prompt else ""
            text = f"{sys_block}{user_prompt}"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def extract_atomic_facts(afg, text, demons, demon_keys, bm25):
    facts = []
    for sent in sentences_from_text(text):
        demos = build_afg_demos_block(demons, demon_keys, bm25, sent)
        system_prompt = f"{AFG_SYSTEM_INSTRUCT}\n{demos}"
        user_prompt = f"Please breakdown the following sentence into independent facts: {sent}"
        out = afg.generate(system_prompt, user_prompt, AFG_MAX_NEW_TOKENS)
        facts.extend(parse_atomic_facts(out))
    return facts


def validate_facts(afv, source, facts, topic):
    results = []
    for fact in facts:
        user_prompt = build_afv_user_prompt(topic, source, fact)
        out = afv.generate(AFV_SYSTEM_INSTRUCT, user_prompt, AFV_MAX_NEW_TOKENS)
        results.append({"fact": fact, "label": parse_afv_label(out), "raw": out})
    return results


def compute_factscore(afg, afv, source, generated, topic, demons, demon_keys, bm25, gamma=10):
    facts = extract_atomic_facts(afg, generated, demons, demon_keys, bm25)
    if not facts:
        return {"n_facts": 0, "n_supported": 0, "n_not_supported": 0, "n_contradicted": 0,
                "init_score": None, "factscore": None, "verified_facts": []}
    verified = validate_facts(afv, source, facts, topic)
    counts = {"SUPPORTED": 0, "NOT_SUPPORTED": 0}
    for v in verified:
        counts[v["label"]] = counts.get(v["label"], 0) + 1
    init_score = counts["SUPPORTED"] / len(facts)
    length_penalty = min(1.0, len(facts) / gamma) if gamma > 0 else 1.0
    return {
        "n_facts": len(facts), "n_supported": counts["SUPPORTED"],
        "n_not_supported": counts["NOT_SUPPORTED"], "n_contradicted": 0,
        "init_score": init_score, "factscore": init_score * length_penalty,
        "verified_facts": verified,
    }


def derive_topic(text, fallback, mode, question=None):
    if mode == "qid":
        return fallback
    if mode == "question" and question:
        return str(question)[:TOPIC_MAX_CHARS]
    if mode == "first-line" and isinstance(text, str):
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line[:TOPIC_MAX_CHARS]
    return fallback


def main():
    ap = argparse.ArgumentParser(description="OpenFActScore on OLMo-32B rewriting chains.")
    ap.add_argument("--input", type=Path, required=True, help="Chain CSV")
    ap.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    ap.add_argument("--afg-model", default=AFG_MODEL_ID)
    ap.add_argument("--afv-model", default=AFV_MODEL_ID)
    ap.add_argument("--topic-mode", choices=["qid", "first-line", "question"], default="qid",
                    help="How to derive the AFV topic from the source row.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qid", action="append", default=None)
    ap.add_argument("--use-4bit", action="store_true",
                    help="Enable 4-bit NF4 for AFG/AFV (needed if AFG=OLMo-3.1-32B).")
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.demos.exists():
        raise FileNotFoundError(args.demos)

    print("=" * 70)
    print(f"OpenFActScore — source-faithfulness against E_0")
    print(f"  AFG: {args.afg_model}")
    print(f"  AFV: {args.afv_model}")
    print(f"  topic-mode: {args.topic_mode}")
    print("=" * 70)

    with open(args.demos) as f:
        demons = json.load(f)
    demon_keys = list(demons.keys())
    bm25 = BM25Okapi([d.split(" ") for d in demon_keys])
    print(f"Loaded {len(demon_keys)} AFG demonstrations")

    df = pd.read_csv(args.input).sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)
    sources = df[df["step"] == 0].set_index(CHAIN_KEYS)["text"].to_dict()
    if not sources:
        raise RuntimeError("No step=0 rows in input CSV.")

    to_eval = df[df["step"] > 0].copy()
    if args.qid:
        to_eval = to_eval[to_eval["qid"].isin(args.qid)]
    if args.limit:
        to_eval = to_eval.head(args.limit)
        print(f"*** SMOKE TEST: first {args.limit} rows ***")

    out_scores = args.input.with_name(args.input.stem + "_openfactscore.csv")
    out_details = args.input.with_name(args.input.stem + "_openfactscore_details.csv")
    print(f"Output scores : {out_scores}")
    print(f"Output details: {out_details}")

    done_keys = set()
    if out_scores.exists():
        prev = pd.read_csv(out_scores)
        done_keys = {tuple(r[k] for k in CHAIN_KEYS + ["step"]) for _, r in prev.iterrows()}
        print(f"Resume: {len(done_keys)} rows already scored")

    afg = HFChatModel(args.afg_model, "AFG", use_4bit=args.use_4bit)
    afv = HFChatModel(args.afv_model, "AFV", use_4bit=args.use_4bit)

    # Cache E_0 facts per qid (for downstream recall analyses).
    e0_facts_cache = {}
    done_e0_qids = set()
    if out_details.exists():
        prev_details = pd.read_csv(out_details)
        done_e0_qids = set(prev_details[prev_details["step"] == 0]["qid"].unique())

    print("\nExtracting E_0 facts ...")
    for chain_id, e0_text in sources.items():
        qid = chain_id[0]
        if qid in e0_facts_cache or qid in done_e0_qids:
            continue
        t0 = time.time()
        facts = extract_atomic_facts(afg, e0_text, demons, demon_keys, bm25)
        e0_facts_cache[qid] = facts
        print(f"  [AFG/E0] {qid}: {len(facts)} facts [{time.time()-t0:.1f}s]", flush=True)
        e0_row = df[(df["qid"] == qid) & (df["step"] == 0)].iloc[0]
        if facts:
            pd.DataFrame([{**{k: e0_row[k] for k in CHAIN_KEYS}, "step": 0,
                           "fact": f, "label": "E0", "raw": ""} for f in facts]).to_csv(
                out_details, mode="a", header=not out_details.exists(),
                index=False, encoding="utf-8",
            )

    total = len(to_eval); t_start = time.time(); n_done = 0

    for i, (_, row) in enumerate(to_eval.iterrows(), start=1):
        chain_id = tuple(row[k] for k in CHAIN_KEYS)
        key = chain_id + (int(row["step"]),)
        if key in done_keys:
            continue
        source = sources.get(chain_id)
        if source is None:
            continue

        topic = derive_topic(source, row["qid"], args.topic_mode, row.get("question"))
        t0 = time.time()
        result = compute_factscore(afg, afv, source, row["text"], topic, demons, demon_keys, bm25)
        elapsed = time.time() - t0

        pd.DataFrame([{**{k: row[k] for k in CHAIN_KEYS},
                       "step": int(row["step"]),
                       "instruction_used": row.get("instruction_used"),
                       **{k: v for k, v in result.items() if k != "verified_facts"}}]
                    ).to_csv(out_scores, mode="a", header=not out_scores.exists(),
                             index=False, encoding="utf-8")

        if result["verified_facts"]:
            pd.DataFrame([{**{k: row[k] for k in CHAIN_KEYS},
                           "step": int(row["step"]),
                           "fact": vf["fact"], "label": vf["label"], "raw": vf["raw"]}
                          for vf in result["verified_facts"]]
                        ).to_csv(out_details, mode="a", header=not out_details.exists(),
                                 index=False, encoding="utf-8")

        n_done += 1
        label = f"{row['group']}/{row['instruction_type']}/run{row['run']}/step{row['step']}"
        if result["factscore"] is None:
            print(f"[{i}/{total}] {label}  no facts  [{elapsed:.1f}s]")
        else:
            print(f"[{i}/{total}] {label}  facts={result['n_facts']:>2} "
                  f"supp={result['n_supported']:>2}  init={result['init_score']:.3f}  "
                  f"score={result['factscore']:.3f}  [{elapsed:.1f}s]", flush=True)
        if i % 10 == 0:
            avg = (time.time() - t_start) / max(n_done, 1)
            print(f"   ETA: {(total - i) * avg / 60:.1f} min  (avg {avg:.1f}s/row)", flush=True)

    print(f"\nTotal: {(time.time() - t_start)/60:.1f} min")
    print(f"Saved: {out_scores}\nSaved: {out_details}")

    out = pd.read_csv(out_scores)
    pivot = out.pivot_table(index="instruction_type", columns="step",
                            values="init_score", aggfunc="median")
    print("\nOpenFActScore — median init_score per (instruction_type, step):")
    print(pivot.round(3))


if __name__ == "__main__":
    main()
