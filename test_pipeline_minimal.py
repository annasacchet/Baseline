#!/usr/bin/env python3
"""
Pipeline test: sample one Q from NewsQA + one from FictionalQA.
Evaluates with:
  - Answer F1 (OLMo-3.1 32B QA model)
  - OpenFactScore (OLMo-2 7B AFG + Gemma-3 4B AFV)

Note: models will be downloaded to HF cache on first run.
Models: OLMo-3.1-32B (QA), OLMo-2-1124-7B (AFG), Gemma-3-4B (AFV)
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
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

REPO_ROOT = Path(__file__).resolve().parent
NEWSQA_DATA = Path("/workspace/Baseline/data/newsqa/combined-newsqa-data-v1.csv")
OUTPUT_DIR = REPO_ROOT / "results" / "test_single_question"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QA_MODEL_ID = "allenai/OLMo-3.1-32B-Instruct"
AFG_MODEL_ID = "allenai/OLMo-2-1124-7B-SFT"
AFV_MODEL_ID = "google/gemma-3-4b-it"

# Download demons.json for AFG few-shot if it exists in repo
DEMONS_PATH = REPO_ROOT / "data" / "demons.json"

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class HFChatModel:
    """Wrapper for HF chat models (AFG, AFV)."""
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


def load_qa_model():
    """Load QA model in 4-bit."""
    print("[*] Loading QA model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        QA_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_ID)
    return model, tokenizer


def load_openfactscore_models():
    """Load AFG (7B) and AFV (4B) models."""
    afg_model = HFChatModel(AFG_MODEL_ID, "AFG", use_4bit=False)
    afv_model = HFChatModel(AFV_MODEL_ID, "AFV", use_4bit=False)
    return afg_model, afv_model


def load_newsqa_sample():
    """Load and sample one question from NewsQA."""
    print(f"[*] Loading NewsQA from {NEWSQA_DATA}...")
    df = pd.read_csv(NEWSQA_DATA)
    print(f"  Loaded {len(df)} rows")

    # Drop rows where validated_answers or story_text is NaN
    df = df.dropna(subset=['validated_answers', 'story_text', 'question'])
    print(f"  After dropping NaN: {len(df)} rows")

    # Sample one random row
    sample = df.sample(n=1).iloc[0]

    return {
        'dataset': 'NewsQA',
        'qid': str(sample['story_id']),
        'question': str(sample['question']),
        'gold_answer': str(sample['validated_answers']),
        'text': str(sample['story_text']),
    }


def load_fictionalqa_sample():
    """Load and sample one question from FictionalQA (from Hugging Face Hub)."""
    print("[*] Loading FictionalQA from HF Hub...")

    from huggingface_hub import hf_hub_download

    # Download parquet file
    joined_path = hf_hub_download(
        repo_id="jwkirchenbauer/fictionalqa",
        filename="joined_qa/data.parquet",
        repo_type="dataset",
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub")
    )

    df = pd.read_parquet(joined_path)

    # Filter to infeasible questions (grade_blind == 0, grade_informed == 1)
    df = df[(df['grade_blind'] == 0) & (df['grade_informed'] == 1)]

    # Remove duplicates
    df = df[df['duplicate_relationship'].isin([None, 'unique']) | (df['is_duplicate_root'] == True)]

    # Sample one
    sample = df.sample(n=1).iloc[0]

    return {
        'dataset': 'FictionalQA',
        'qid': sample['fiction_id'],
        'question': sample['question'],
        'gold_answer': sample['natural_answer'],
        'text': sample['document'],
    }


def normalize_answer(s):
    """Lower text and remove articles, punctuation, extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, gold):
    """Compute F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if len(prediction_tokens) == 0 or len(gold_tokens) == 0:
        return int(prediction_tokens == gold_tokens)
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def run_qa(text, question, model, tokenizer):
    """Run QA model on text + question."""
    prompt = f"""Answer the following question based on the provided text.

Text:
{text[:2000]}

Question: {question}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=96,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("Answer:")[-1].strip()
    return answer


# ============================================================================
# OpenFactScore Functions
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
    """Split text into sentences, handling edge cases."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences = []
    for para in paragraphs:
        curr = sent_tokenize(para)
        sentences.extend(curr)
    return sentences[:20]  # Limit to first 20 sentences for test


def parse_atomic_facts(generated_text):
    """Parse AFG output into atomic facts."""
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
    """Build AFV user prompt."""
    definition = f"Answer the question about {topic} based on the given context.\n\n"
    context = f"Title: {topic}\nText: {source.strip()}\n\n"
    definition += context.strip()
    if definition[-1] not in string.punctuation:
        definition += "."
    return f"{definition.strip()}\n\nInput: {claim.strip()} True or False?\nAnswer:"


def parse_afv_label(generated_text):
    """Parse AFV output (True/False)."""
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


def extract_atomic_facts(afg_model, text):
    """Extract atomic facts from text using AFG."""
    sentences = sentences_from_text(text)
    all_facts = []
    for sent in sentences:
        user_prompt = f"Please breakdown the following sentence into independent facts: {sent}"
        out = afg_model.generate(AFG_SYSTEM_INSTRUCT, user_prompt, 256)
        all_facts.extend(parse_atomic_facts(out))
    return all_facts


def validate_facts(afv_model, source, facts, topic):
    """Validate facts against source using AFV."""
    results = []
    for fact in facts:
        user_prompt = build_afv_prompt(topic, source, fact)
        out = afv_model.generate(AFV_SYSTEM_INSTRUCT, user_prompt, 8)
        label = parse_afv_label(out)
        results.append({"fact": fact, "label": label})
    return results


def compute_openfactscore(afg_model, afv_model, text, topic, gamma=10):
    """Compute OpenFactScore for text."""
    facts = extract_atomic_facts(afg_model, text)
    if not facts:
        return {
            "n_facts": 0,
            "n_supported": 0,
            "n_not_supported": 0,
            "factscore": None,
        }

    verified = validate_facts(afv_model, text, facts, topic)
    counts = {"SUPPORTED": 0, "NOT_SUPPORTED": 0}
    for v in verified:
        counts[v["label"]] = counts.get(v["label"], 0) + 1

    init_score = counts["SUPPORTED"] / len(facts)
    length_penalty = min(1.0, len(facts) / gamma) if gamma > 0 else 1.0
    final_score = init_score * length_penalty

    return {
        "n_facts": len(facts),
        "n_supported": counts["SUPPORTED"],
        "n_not_supported": counts["NOT_SUPPORTED"],
        "init_score": init_score,
        "factscore": final_score,
    }


def main():
    print("=" * 70)
    print("Pipeline Test: NewsQA + FictionalQA")
    print("Metrics: Answer F1 + OpenFactScore")
    print("=" * 70)

    print("\n[*] Loading models...")
    qa_model, qa_tokenizer = load_qa_model()
    afg_model, afv_model = load_openfactscore_models()
    results = []

    # NewsQA
    print("\n[*] Testing NewsQA...")
    newsqa = load_newsqa_sample()
    print(f"  Question: {newsqa['question'][:80]}...")
    print(f"  Gold answer: {newsqa['gold_answer'][:60]}...")

    answer_newsqa = run_qa(newsqa['text'], newsqa['question'], qa_model, qa_tokenizer)
    f1_newsqa = f1_score(answer_newsqa, newsqa['gold_answer'])
    print(f"  Predicted: {answer_newsqa[:60]}...")
    print(f"  Answer F1: {f1_newsqa:.3f}")

    print(f"  Computing OpenFactScore...")
    topic_newsqa = newsqa['text'].split('\n')[0][:200]
    ofscore_newsqa = compute_openfactscore(afg_model, afv_model, newsqa['text'], topic_newsqa)
    print(f"  OpenFactScore: {ofscore_newsqa['factscore']:.3f} ({ofscore_newsqa['n_supported']}/{ofscore_newsqa['n_facts']} facts)")

    results.append({
        'dataset': 'NewsQA',
        'qid': newsqa['qid'],
        'question': newsqa['question'][:100],
        'gold_answer': newsqa['gold_answer'][:50],
        'predicted_answer': answer_newsqa[:100],
        'answer_f1': f1_newsqa,
        'n_facts': ofscore_newsqa['n_facts'],
        'n_supported': ofscore_newsqa['n_supported'],
        'factscore': ofscore_newsqa['factscore'],
    })

    # FictionalQA
    print("\n[*] Testing FictionalQA...")
    fictionalqa = load_fictionalqa_sample()
    print(f"  Question: {fictionalqa['question'][:80]}...")
    print(f"  Gold answer: {fictionalqa['gold_answer'][:60]}...")

    answer_fictionalqa = run_qa(fictionalqa['text'], fictionalqa['question'], qa_model, qa_tokenizer)
    f1_fictionalqa = f1_score(answer_fictionalqa, fictionalqa['gold_answer'])
    print(f"  Predicted: {answer_fictionalqa[:60]}...")
    print(f"  Answer F1: {f1_fictionalqa:.3f}")

    print(f"  Computing OpenFactScore...")
    topic_fictionalqa = fictionalqa['text'].split('\n')[0][:200]
    ofscore_fictionalqa = compute_openfactscore(afg_model, afv_model, fictionalqa['text'], topic_fictionalqa)
    print(f"  OpenFactScore: {ofscore_fictionalqa['factscore']:.3f} ({ofscore_fictionalqa['n_supported']}/{ofscore_fictionalqa['n_facts']} facts)")

    results.append({
        'dataset': 'FictionalQA',
        'qid': fictionalqa['qid'],
        'question': fictionalqa['question'][:100],
        'gold_answer': fictionalqa['gold_answer'][:50],
        'predicted_answer': answer_fictionalqa[:100],
        'answer_f1': f1_fictionalqa,
        'n_facts': ofscore_fictionalqa['n_facts'],
        'n_supported': ofscore_fictionalqa['n_supported'],
        'factscore': ofscore_fictionalqa['factscore'],
    })

    # Save results
    results_df = pd.DataFrame(results)
    output_csv = OUTPUT_DIR / "test_pipeline_results.csv"
    results_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 70)
    print(f"[✓] Test completed! Results: {output_csv}")
    print("=" * 70)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
