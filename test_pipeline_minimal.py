#!/usr/bin/env python3
"""
Pipeline test: sample one Q from NewsQA + one from FictionalQA.
Evaluates with:
  - Answer F1 (OLMo-3.1 32B QA model)
  - OpenFactScore (OLMo-2 7B AFG + Gemma-3 4B AFV)

Mirrors the logic of the production scripts:
  - NewsQA: parses validated_answers / answer_char_ranges into real text spans
  - QA: uses apply_chat_template and decodes only new tokens (no "assistant\\n" noise)
  - F1: max over (gold + aliases)
"""

import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent
NEWSQA_DATA = Path("/workspace/Baseline/data/newsqa/combined-newsqa-data-v1.csv")
OUTPUT_DIR = REPO_ROOT / "results" / "test_single_question"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QA_MODEL_ID = "allenai/OLMo-3.1-32B-Instruct"
AFG_MODEL_ID = "allenai/OLMo-2-1124-7B-SFT"
AFV_MODEL_ID = "google/gemma-3-4b-it"

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ============================================================================
# HFChatModel wrapper (used by AFG/AFV/QA)
# ============================================================================

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
            kwargs["dtype"] = torch.bfloat16
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
        # Decode ONLY new tokens — strips the "assistant\n" prefix
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================================
# NewsQA: parse char-ranges from validated_answers / answer_char_ranges
# ============================================================================

def _span_to_text(span_key: str, story_text: str):
    """Convert '294:297' or '10:20,30:40' to the actual answer text."""
    parts = []
    try:
        for span in span_key.split(","):
            s_str, e_str = span.split(":")
            s, e = int(s_str), int(e_str)
            parts.append(story_text[s:e])
    except (ValueError, IndexError):
        return None
    out = " ".join(p.strip() for p in parts if p.strip()).strip()
    return out or None


def parse_validated_answers(value, story_text):
    """Returns list of (answer_text, vote_count)."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        d = json.loads(value)
    except json.JSONDecodeError:
        return []
    answers = []
    for key, count in d.items():
        if key in ("none", "badQuestion"):
            continue
        ans = _span_to_text(key, story_text)
        if ans:
            answers.append((ans, int(count)))
    answers.sort(key=lambda t: -t[1])
    return answers


def parse_sourcer_agreement(value, story_text):
    """Returns list of (answer_text, vote_count) from |-separated answer_char_ranges."""
    if not isinstance(value, str) or not value.strip():
        return []
    counts = {}
    for raw in value.split("|"):
        raw = raw.strip()
        if not raw or raw == "None":
            continue
        ans = _span_to_text(raw, story_text)
        if ans:
            counts[ans] = counts.get(ans, 0) + 1
    return sorted(counts.items(), key=lambda t: -t[1])


def load_newsqa_sample():
    """Load NewsQA, parse char-ranges, return a usable Q with gold answer text + aliases."""
    print(f"[*] Loading NewsQA from {NEWSQA_DATA}...")
    df = pd.read_csv(NEWSQA_DATA)
    print(f"  Loaded {len(df)} rows")

    df = df.dropna(subset=['story_text', 'question'])

    # Try to find a row where we can extract a real answer text from char-ranges
    for _ in range(100):  # try up to 100 random rows
        sample = df.sample(n=1).iloc[0]
        story_text = str(sample['story_text'])

        # Try validated_answers first
        v_spans = parse_validated_answers(sample.get('validated_answers'), story_text)
        v_agreed = [(a, c) for a, c in v_spans if c >= 2]

        if v_agreed:
            answers = sorted({a for a, _ in v_agreed})
            return {
                'dataset': 'NewsQA',
                'qid': str(sample['story_id']),
                'question': str(sample['question']),
                'gold_answer': v_agreed[0][0],
                'aliases': answers,
                'text': story_text,
            }

        # Fall back to sourcer agreement
        s_spans = parse_sourcer_agreement(sample.get('answer_char_ranges'), story_text)
        s_agreed = [(a, c) for a, c in s_spans if c >= 2]
        if s_agreed:
            answers = sorted({a for a, _ in s_agreed})
            return {
                'dataset': 'NewsQA',
                'qid': str(sample['story_id']),
                'question': str(sample['question']),
                'gold_answer': s_agreed[0][0],
                'aliases': answers,
                'text': story_text,
            }

    raise RuntimeError("Couldn't find an answerable NewsQA question after 100 tries")


def load_fictionalqa_sample():
    """Load FictionalQA from HF Hub."""
    print("[*] Loading FictionalQA from HF Hub...")
    from huggingface_hub import hf_hub_download

    qa_path = hf_hub_download(
        repo_id="jwkirchenbauer/fictionalqa",
        filename="joined_qa/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(qa_path)
    print(f"  Loaded {len(df)} rows")

    df = df[(df['grade_blind'] == 0) & (df['grade_informed'] == 1)]
    df = df.dropna(subset=['natural_answer', 'fiction', 'question'])
    print(f"  After filtering: {len(df)} rows")

    sample = df.sample(n=1).iloc[0]
    nat = str(sample['natural_answer']).strip()
    aliases = [nat]
    span = sample.get('span_answer')
    if span and isinstance(span, str) and span.strip() and span.strip() != nat:
        aliases.append(span.strip())

    return {
        'dataset': 'FictionalQA',
        'qid': str(sample['fiction_id']),
        'question': str(sample['question']),
        'gold_answer': nat,
        'aliases': aliases,
        'text': str(sample['fiction']),
    }


# ============================================================================
# Answer F1 (SQuAD-style)
# ============================================================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split() if a_gold else []
    pred_toks = normalize_answer(a_pred).split() if a_pred else []
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def best_f1(pred, golds):
    """Max F1 over all gold answer strings."""
    return max((compute_f1(g, pred) for g in golds), default=0.0)


# ============================================================================
# QA evaluation
# ============================================================================

QA_USER_TEMPLATE = """Answer the question based on the context below. Give a short, direct answer — a few words at most, no explanation.

Context:
{context}

Question: {question}
Answer:"""


def run_qa(text, question, qa_model_wrapper):
    """Run QA using the wrapper (handles chat template + new-tokens-only decode)."""
    user_prompt = QA_USER_TEMPLATE.format(context=text.strip(), question=question.strip())
    return qa_model_wrapper.generate(system_prompt=None, user_prompt=user_prompt, max_new_tokens=96)


# ============================================================================
# OpenFactScore
# ============================================================================

AFG_SYSTEM_INSTRUCT = (
    "You are an annotator that breaks down sentences into "
    "independent facts, short statements that each contain one piece of "
    "information contained in the given sentence. "
    "Do not add new entities, do not deviate from the subject "
    "of the sentence given by the user, do not hallucinate. "
    "List the sentences using -"
)

AFV_SYSTEM_INSTRUCT = (
    "You are an annotator that verifies the factuality of a sentence "
    "according to a given source text. You answer only True or False and "
    "provides no further explanations."
)


def sentences_from_text(text):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences = []
    for para in paragraphs:
        sentences.extend(sent_tokenize(para))
    return sentences[:20]  # cap at 20 for the test


def parse_atomic_facts(generated_text):
    text = generated_text.replace("<|eot_id|>", "")
    text = re.sub(r"-\s*\n", "", text)
    facts = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            fact = line[2:].strip()
            if fact and fact[-1] not in string.punctuation:
                fact = fact + "."
            if fact:
                facts.append(fact)
        elif not line:
            if facts:
                break
        else:
            if facts:
                break
    return facts


def build_afv_prompt(topic, source, claim):
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
            return "SUPPORTED"
        elif "false" in answer and "true" not in answer:
            return "NOT_SUPPORTED"
        else:
            return "SUPPORTED" if answer.index("true") > answer.index("false") else "NOT_SUPPORTED"
    stripped = answer.translate(str.maketrans("", "", string.punctuation)).split()
    return "SUPPORTED" if all(kw not in stripped for kw in ("not", "cannot", "unknown", "information")) else "NOT_SUPPORTED"


def extract_atomic_facts(afg_model, text):
    sentences = sentences_from_text(text)
    all_facts = []
    for sent in sentences:
        user_prompt = f"Please breakdown the following sentence into independent facts: {sent}"
        out = afg_model.generate(AFG_SYSTEM_INSTRUCT, user_prompt, 256)
        all_facts.extend(parse_atomic_facts(out))
    return all_facts


def validate_facts(afv_model, source, facts, topic):
    results = []
    for fact in facts:
        user_prompt = build_afv_prompt(topic, source, fact)
        out = afv_model.generate(AFV_SYSTEM_INSTRUCT, user_prompt, 8)
        results.append({"fact": fact, "label": parse_afv_label(out)})
    return results


def compute_openfactscore(afg_model, afv_model, text, topic, gamma=10):
    facts = extract_atomic_facts(afg_model, text)
    if not facts:
        return {"n_facts": 0, "n_supported": 0, "factscore": None}
    verified = validate_facts(afv_model, text, facts, topic)
    n_supported = sum(1 for v in verified if v["label"] == "SUPPORTED")
    init_score = n_supported / len(facts)
    length_penalty = min(1.0, len(facts) / gamma) if gamma > 0 else 1.0
    return {
        "n_facts": len(facts),
        "n_supported": n_supported,
        "init_score": init_score,
        "factscore": init_score * length_penalty,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Pipeline Test: NewsQA + FictionalQA")
    print("Metrics: Answer F1 + OpenFactScore")
    print("=" * 70)

    print("\n[*] Loading models...")
    qa_model = HFChatModel(QA_MODEL_ID, "QA", use_4bit=True)
    afg_model = HFChatModel(AFG_MODEL_ID, "AFG", use_4bit=False)
    afv_model = HFChatModel(AFV_MODEL_ID, "AFV", use_4bit=False)
    results = []

    for loader_fn, name in [(load_newsqa_sample, "NewsQA"), (load_fictionalqa_sample, "FictionalQA")]:
        print(f"\n[*] Testing {name}...")
        sample = loader_fn()
        print(f"  Question: {sample['question']}")
        print(f"  Gold answer: {sample['gold_answer']}")
        print(f"  Aliases: {sample['aliases']}")
        print(f"  Text length: {len(sample['text'])} chars")

        answer = run_qa(sample['text'], sample['question'], qa_model)
        f1 = best_f1(answer, sample['aliases'])
        print(f"  Predicted: {answer}")
        print(f"  Answer F1: {f1:.3f}")

        print(f"  Computing OpenFactScore...")
        topic = sample['text'].split('\n')[0][:200]
        ofs = compute_openfactscore(afg_model, afv_model, sample['text'], topic)
        print(f"  OpenFactScore: {ofs['factscore']:.3f} ({ofs['n_supported']}/{ofs['n_facts']} facts)")

        results.append({
            'dataset': sample['dataset'],
            'qid': sample['qid'],
            'question': sample['question'][:100],
            'gold_answer': sample['gold_answer'][:60],
            'predicted_answer': answer[:100],
            'answer_f1': f1,
            'n_facts': ofs['n_facts'],
            'n_supported': ofs['n_supported'],
            'factscore': ofs['factscore'],
        })

    results_df = pd.DataFrame(results)
    output_csv = OUTPUT_DIR / "test_pipeline_results.csv"
    results_df.to_csv(output_csv, index=False)
    print("\n" + "=" * 70)
    print(f"[✓] Results: {output_csv}")
    print("=" * 70)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
