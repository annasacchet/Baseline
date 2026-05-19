"""
OpenFActScore — RECALL for 300q.

Phase 1 (AFG): extract atomic facts from the 297 unique E_0 (step=0) texts.
Phase 2 (AFV): for each rewritten step E_k, verify each E_0 fact against E_k.

recall_init = n_recalled / n_e0_facts
recall      = recall_init * min(1, n_e0_facts / gamma)   [length penalty]

Output:
  results/300q/rewriting_chains_300q_e0_facts.csv
    qid, fact

  results/300q/rewriting_chains_300q_openfactscore_recall.csv
    qid, group, instruction_type, run, step, instruction_used,
    n_e0_facts, n_recalled, n_not_recalled, recall_init, recall

  results/300q/rewriting_chains_300q_openfactscore_recall_details.csv
    qid, group, instruction_type, run, step, fact, label, raw
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = REPO_ROOT / "results" / "300q" / "rewriting_chains_300q.csv"
DEFAULT_DEMOS = REPO_ROOT / "data" / "demons.json"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]

AFG_MODEL_ID = "allenai/OLMo-2-1124-7B-SFT"
AFV_MODEL_ID = "google/gemma-3-4b-it"

K_BM25 = 1
AFG_MAX_NEW_TOKENS = 256
AFV_MAX_NEW_TOKENS = 8

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    return [m.group() for m in re.finditer(pattern, text)]


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
# AFG
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


# ---------------------------------------------------------------------------
# AFV
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model wrapper
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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def extract_atomic_facts(afg, text, demons, demon_keys, bm25):
    sentences = sentences_from_text(text)
    all_facts = []
    for sent in sentences:
        demos_block = build_afg_demos_block(demons, demon_keys, bm25, sent)
        system_prompt = f"{AFG_SYSTEM_INSTRUCT}\n{demos_block}"
        user_prompt = f"Please breakdown the following sentence into independent facts: {sent}"
        out = afg.generate(system_prompt, user_prompt, AFG_MAX_NEW_TOKENS)
        all_facts.extend(parse_atomic_facts(out))
    return all_facts


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OFS Recall — AFG on E_0 then AFV against each E_k.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--afg-model", default=AFG_MODEL_ID)
    parser.add_argument("--afv-model", default=AFV_MODEL_ID)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test: process only first N qids end-to-end.")
    parser.add_argument("--qid", action="append", default=None)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Chain CSV not found: {args.input}")
    if not args.demos.exists():
        raise FileNotFoundError(f"demons.json not found: {args.demos}")

    print("=" * 70)
    print("OpenFActScore RECALL — AFG on E_0, AFV against E_k")
    print(f"  AFG: {args.afg_model}  AFV: {args.afv_model}  4-bit={args.use_4bit}")
    print("=" * 70)

    with open(args.demos) as f:
        demons = json.load(f)
    demon_keys = list(demons.keys())
    bm25 = BM25Okapi([doc.split(" ") for doc in demon_keys])
    print(f"Loaded {len(demon_keys)} AFG demonstrations")

    df = pd.read_csv(args.input)
    df = df.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    e0_unique = df[df["step"] == 0][["qid", "text"]].drop_duplicates("qid").set_index("qid")["text"].to_dict()
    print(f"Found {len(e0_unique)} unique E_0 texts")

    to_eval = df[df["step"] > 0].copy()
    if args.qid:
        to_eval = to_eval[to_eval["qid"].isin(args.qid)]
        e0_unique = {k: v for k, v in e0_unique.items() if k in args.qid}
    if args.limit:
        qids_limited = list(e0_unique.keys())[:args.limit]
        e0_unique = {k: v for k, v in e0_unique.items() if k in qids_limited}
        to_eval = to_eval[to_eval["qid"].isin(qids_limited)]
        print(f"*** SMOKE TEST: first {args.limit} qids → {len(to_eval)} rows ***")

    out_facts   = args.input.with_name(args.input.stem.replace("rewriting_chains_", "") + "_e0_facts.csv")
    out_scores  = args.input.with_name(args.input.stem + "_openfactscore_recall.csv")
    out_details = args.input.with_name(args.input.stem + "_openfactscore_recall_details.csv")
    print(f"E_0 facts:      {out_facts}")
    print(f"Recall scores:  {out_scores}")
    print(f"Recall details: {out_details}")

    # Load already-extracted E_0 facts (resumability)
    e0_facts_cache = {}
    if out_facts.exists():
        prev_facts = pd.read_csv(out_facts)
        for qid, grp in prev_facts.groupby("qid"):
            e0_facts_cache[qid] = grp["fact"].tolist()
        print(f"Resume: E_0 facts already extracted for {len(e0_facts_cache)} qids.")

    # Load already-scored (chain, step) pairs (resumability)
    done_keys = set()
    if out_scores.exists():
        prev_scores = pd.read_csv(out_scores)
        done_keys = {tuple(r[k] for k in CHAIN_KEYS + ["step"]) for _, r in prev_scores.iterrows()}
        print(f"Resume: {len(done_keys)} (chain, step) already scored.")

    afg = HFChatModel(args.afg_model, "AFG", use_4bit=args.use_4bit)
    afv = HFChatModel(args.afv_model, "AFV", use_4bit=args.use_4bit)

    # -----------------------------------------------------------------------
    # Phase 1 — AFG on E_0
    # -----------------------------------------------------------------------
    qids_to_extract = [q for q in e0_unique if q not in e0_facts_cache]
    print(f"\n--- Phase 1: AFG on {len(qids_to_extract)} E_0 texts ---")
    t_phase1 = time.time()
    for i, qid in enumerate(qids_to_extract, 1):
        print(f"[{i}/{len(qids_to_extract)}] {qid} ...", end=" ", flush=True)
        t0 = time.time()
        facts = extract_atomic_facts(afg, e0_unique[qid], demons, demon_keys, bm25)
        e0_facts_cache[qid] = facts
        print(f"{len(facts)} facts [{time.time()-t0:.1f}s]", flush=True)
        if facts:
            pd.DataFrame({"qid": qid, "fact": facts}).to_csv(
                out_facts, mode="a", header=not out_facts.exists(), index=False, encoding="utf-8"
            )
    print(f"Phase 1 done in {(time.time()-t_phase1)/60:.1f} min")

    # -----------------------------------------------------------------------
    # Phase 2 — AFV: verify E_0 facts against each E_k
    # -----------------------------------------------------------------------
    total = len(to_eval)
    print(f"\n--- Phase 2: AFV on {total} rewritten steps ---")
    t_phase2 = time.time()
    n_done = 0

    for i, (_, row) in enumerate(to_eval.iterrows(), start=1):
        chain_id = tuple(row[k] for k in CHAIN_KEYS)
        key = chain_id + (int(row["step"]),)
        if key in done_keys:
            continue

        e0_facts = e0_facts_cache.get(row["qid"], [])
        label_str = f"{row['group']}/{row['instruction_type']}/run{int(row['run'])}/step{int(row['step'])}"
        print(f"[{i}/{total}] {label_str} ({len(e0_facts)} facts) ...", end=" ", flush=True)
        t0 = time.time()

        result = compute_recall(afv, e0_facts, row["text"], row["qid"])
        elapsed = time.time() - t0

        score_row = pd.DataFrame([{
            **{k: row[k] for k in CHAIN_KEYS},
            "step": int(row["step"]),
            "instruction_used": row.get("instruction_used"),
            **{k: v for k, v in result.items() if k != "verified_facts"},
        }])
        score_row.to_csv(out_scores, mode="a", header=not out_scores.exists(), index=False, encoding="utf-8")

        if result["verified_facts"]:
            pd.DataFrame([
                {**{k: row[k] for k in CHAIN_KEYS},
                 "step": int(row["step"]), "fact": vf["fact"],
                 "label": vf["label"], "raw": vf["raw"]}
                for vf in result["verified_facts"]
            ]).to_csv(out_details, mode="a", header=not out_details.exists(), index=False, encoding="utf-8")

        n_done += 1
        if result["recall_init"] is None:
            print(f"no facts [{elapsed:.1f}s]")
        else:
            print(
                f"recalled={result['n_recalled']:>2}/{result['n_e0_facts']:>2}  "
                f"recall_init={result['recall_init']:.3f}  [{elapsed:.1f}s]",
                flush=True,
            )

        if i % 10 == 0:
            avg = (time.time() - t_phase2) / max(n_done, 1)
            print(f"   ETA: {(total - i) * avg / 60:.1f} min  (avg {avg:.1f}s/row)", flush=True)

    print(f"\nPhase 2 done in {(time.time()-t_phase2)/60:.1f} min")
    print(f"Saved: {out_scores}")
    print(f"Saved: {out_details}")

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
