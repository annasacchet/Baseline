"""
FactScore-style evaluation (source faithfulness variant).

Atomic fact extraction: canonical pipeline from Min et al. (2023) — sentence
tokenization + BM25 few-shot using the official demons.json demonstrations.
Verification: claim-vs-source NLI done by GPT-4o-mini against the step-0 text
of each chain (E0), instead of Wikipedia retrieval.
"""

import json
import os
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from rank_bm25 import BM25Okapi

CSV_PATH = "/Users/annasacchet/Desktop/RISULTATI TEST/rewriting_chains32b.csv"
DEMOS_PATH = "/Users/annasacchet/.cache/factscore/demos/demons.json"
JUDGE_MODEL = "gpt-4o-mini"
CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]
MAX_RETRIES = 6
RETRY_BACKOFF = 1.0

N_FIXED_DEMOS = 7
K_BM25 = 1

MAX_WORKERS = 4

TEST_MODE = True
TEST_N_CHAINS = 1
TEST_FILTER = {"group": "style", "instruction_type": "paraphrase", "run": 0}

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    return [m.group() for m in re.finditer(pattern, text)]


def fix_sentence_splitter(sentences, initials):
    for initial in initials:
        if not np_any(initial in s for s in sentences):
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


def np_any(it):
    for v in it:
        if v:
            return True
    return False


def sentences_from_text(text):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences = []
    for para in paragraphs:
        initials = detect_initials(para)
        curr = sent_tokenize(para)
        curr = fix_sentence_splitter(curr, initials)
        sentences.extend(curr)
    return sentences


def build_extraction_prompt(sentence, demons, demon_keys, bm25):
    prompt_parts = []
    for i in range(N_FIXED_DEMOS):
        key = demon_keys[i]
        prompt_parts.append(
            f"Please breakdown the following sentence into independent facts: {key}"
        )
        for fact in demons[key]:
            prompt_parts.append(f"- {fact}")
        prompt_parts.append("")

    tokenized_query = sentence.split(" ")
    top_matches = bm25.get_top_n(tokenized_query, demon_keys, K_BM25)
    for match in top_matches:
        prompt_parts.append(
            f"Please breakdown the following sentence into independent facts: {match}"
        )
        for fact in demons[match]:
            prompt_parts.append(f"- {fact}")
        prompt_parts.append("")

    prompt_parts.append(
        f"Please breakdown the following sentence into independent facts: {sentence}"
    )
    return "\n".join(prompt_parts)


def parse_extraction_output(text):
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
    return facts


def _parse_retry_after(err_msg):
    match = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", err_msg)
    if not match:
        return None
    value = float(match.group(1))
    return value / 1000.0 if match.group(2) == "ms" else value


def call_with_retries(client, **kwargs):
    import random
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            err_str = str(e)
            is_rate_limit = "rate_limit" in err_str.lower() or "429" in err_str
            suggested = _parse_retry_after(err_str)

            if is_rate_limit and suggested is not None:
                wait = max(suggested + 1.0, RETRY_BACKOFF * (2 ** attempt))
            elif is_rate_limit:
                wait = RETRY_BACKOFF * (2 ** attempt) + random.uniform(0, 2)
            else:
                wait = RETRY_BACKOFF * (attempt + 1)

            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
    raise RuntimeError(f"OpenAI call failed after {MAX_RETRIES} attempts: {last_err}")


def _extract_one_sentence(client, sent, demons, demon_keys, bm25):
    if sent.lower().startswith(("sure", "here are", "please", "i hope")):
        return []
    prompt = build_extraction_prompt(sent, demons, demon_keys, bm25)
    content = call_with_retries(
        client,
        model=JUDGE_MODEL,
        temperature=0.0,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return parse_extraction_output(content)


def extract_atomic_facts(client, text, demons, demon_keys, bm25, executor):
    sentences = sentences_from_text(text)
    results = list(
        executor.map(
            lambda s: _extract_one_sentence(client, s, demons, demon_keys, bm25),
            sentences,
        )
    )
    all_facts = []
    for r in results:
        all_facts.extend(r)
    return all_facts


VERIFICATION_USER_TEMPLATE = """Answer the question about {topic} based on the given context.

{source}

Input: {claim} True, False, or Contradicted?
Output a single JSON object with this exact schema (no extra text):
{{"label": "SUPPORTED" | "NOT_SUPPORTED" | "CONTRADICTED", "reason": "short justification"}}

Use SUPPORTED if the context explicitly states or directly entails the claim (paraphrases with the same meaning count as SUPPORTED).
Use NOT_SUPPORTED if the context neither states nor entails the claim.
Use CONTRADICTED if the context explicitly contradicts the claim."""


def verify_fact(client, source, claim, topic):
    content = call_with_retries(
        client,
        model=JUDGE_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": VERIFICATION_USER_TEMPLATE.format(
                    topic=topic, source=source.strip(), claim=claim
                ),
            },
        ],
    )
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {"label": "NOT_SUPPORTED", "reason": "parse_error"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"label": "NOT_SUPPORTED", "reason": "parse_error"}
    label = data.get("label", "NOT_SUPPORTED").upper()
    if label not in {"SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"}:
        label = "NOT_SUPPORTED"
    return {"label": label, "reason": data.get("reason", "")}


def compute_factscore(client, source, generated, topic, demons, demon_keys, bm25, executor, gamma=10):
    facts = extract_atomic_facts(client, generated, demons, demon_keys, bm25, executor)
    if not facts:
        return {
            "n_facts": 0,
            "n_supported": 0,
            "n_not_supported": 0,
            "n_contradicted": 0,
            "init_score": None,
            "factscore": None,
            "verified_facts": [],
        }

    verify_results = list(
        executor.map(lambda f: verify_fact(client, source, f, topic), facts)
    )
    verified = []
    counts = {"SUPPORTED": 0, "NOT_SUPPORTED": 0, "CONTRADICTED": 0}
    for fact, res in zip(facts, verify_results):
        counts[res["label"]] += 1
        verified.append({"fact": fact, **res})

    init_score = counts["SUPPORTED"] / len(facts)
    length_penalty = min(1.0, len(facts) / gamma) if gamma > 0 else 1.0
    final_score = init_score * length_penalty

    return {
        "n_facts": len(facts),
        "n_supported": counts["SUPPORTED"],
        "n_not_supported": counts["NOT_SUPPORTED"],
        "n_contradicted": counts["CONTRADICTED"],
        "init_score": init_score,
        "factscore": final_score,
        "verified_facts": verified,
    }


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY non è settata. Esegui:\n  export OPENAI_API_KEY='sk-...'")

    path = Path(CSV_PATH)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    if not Path(DEMOS_PATH).exists():
        raise FileNotFoundError(f"demons.json non trovato: {DEMOS_PATH}")

    with open(DEMOS_PATH) as f:
        demons = json.load(f)
    demon_keys = list(demons.keys())
    tokenized_corpus = [doc.split(" ") for doc in demon_keys]
    bm25 = BM25Okapi(tokenized_corpus)

    client = OpenAI()
    df = pd.read_csv(path)
    df = df.sort_values(CHAIN_KEYS + ["step"]).reset_index(drop=True)

    sources = df[df["step"] == 0].set_index(CHAIN_KEYS)["text"].to_dict()
    if not sources:
        raise RuntimeError("Nessun testo con step=0 trovato.")

    to_eval = df[df["step"] > 0]

    if TEST_MODE:
        pool = to_eval
        if TEST_FILTER:
            for k, v in TEST_FILTER.items():
                pool = pool[pool[k] == v]
        chain_tuples = pool[CHAIN_KEYS].drop_duplicates().head(TEST_N_CHAINS)
        if chain_tuples.empty:
            raise RuntimeError(f"Nessuna chain corrisponde al filtro: {TEST_FILTER}")
        mask = False
        for _, ct in chain_tuples.iterrows():
            m = True
            for k in CHAIN_KEYS:
                m = m & (to_eval[k] == ct[k])
            mask = mask | m
        to_eval = to_eval[mask]
        filter_desc = f" filtro={TEST_FILTER}" if TEST_FILTER else ""
        print(f"*** TEST MODE: {TEST_N_CHAINS} chain(s){filter_desc} ***")

    total = len(to_eval)
    print(f"FactScore su {total} testi (step > 0) con judge = {JUDGE_MODEL}")
    print(f"Source = step 0 della chain ({len(sources)} chains disponibili)")
    print(f"Extraction: few-shot canonico Min et al. (2023), {N_FIXED_DEMOS} fixed + {K_BM25} BM25 demos")
    print(f"Parallelismo: {MAX_WORKERS} thread")
    print()

    results, details = [], []
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    t_start = time.time()
    for i, (_, row) in enumerate(to_eval.iterrows(), start=1):
        chain_id = tuple(row[k] for k in CHAIN_KEYS)
        source = sources.get(chain_id)
        if source is None:
            continue

        label = f"{row['group']}/{row['instruction_type']}/run{row['run']}/step{row['step']}"
        t0 = time.time()
        print(f"[{i}/{total}] {label} ...", end=" ", flush=True)

        result = compute_factscore(
            client, source, row["text"], row["qid"], demons, demon_keys, bm25, executor
        )
        elapsed = time.time() - t0

        row_out = {
            **{k: row[k] for k in CHAIN_KEYS},
            "step": int(row["step"]),
            "instruction_used": row.get("instruction_used"),
            **{k: v for k, v in result.items() if k != "verified_facts"},
        }
        results.append(row_out)
        for vf in result["verified_facts"]:
            details.append({
                **{k: row[k] for k in CHAIN_KEYS},
                "step": int(row["step"]),
                **vf,
            })

        if result["factscore"] is None:
            print(f"no facts  [{elapsed:.1f}s]")
        else:
            print(
                f"facts={result['n_facts']:>2}  "
                f"supp={result['n_supported']:>2}  "
                f"contr={result['n_contradicted']:>2}  "
                f"init={result['init_score']:.3f}  "
                f"score={result['factscore']:.3f}  "
                f"[{elapsed:.1f}s]"
            )

    executor.shutdown(wait=True)
    total_elapsed = time.time() - t_start
    print(f"\nTempo totale: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    results_df = pd.DataFrame(results)
    details_df = pd.DataFrame(details)

    out_scores = path.with_name(path.stem + "_factscore.csv")
    out_details = path.with_name(path.stem + "_factscore_details.csv")

    if out_scores.exists():
        prev_scores = pd.read_csv(out_scores)
        merged = pd.concat([prev_scores, results_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=CHAIN_KEYS + ["step"], keep="last")
        merged.to_csv(out_scores, index=False)
    else:
        results_df.to_csv(out_scores, index=False)

    if out_details.exists():
        prev_details = pd.read_csv(out_details)
        merged_d = pd.concat([prev_details, details_df], ignore_index=True)
        merged_d = merged_d.drop_duplicates(
            subset=CHAIN_KEYS + ["step", "fact"], keep="last"
        )
        merged_d.to_csv(out_details, index=False)
    else:
        details_df.to_csv(out_details, index=False)

    all_scores = pd.read_csv(out_scores)

    print()
    print("=" * 70)
    print("FACTSCORE PER CHAIN (righe=chain, colonne=step) — TUTTE le run finora")
    print("=" * 70)
    pivot = all_scores.pivot_table(
        index=["group", "instruction_type", "run"],
        columns="step",
        values="factscore",
        aggfunc="first",
    )
    print(pivot.round(3))

    print()
    print(f"Scores salvati in: {out_scores}")
    print(f"Dettaglio fatti in: {out_details}")

    return results_df, details_df


if __name__ == "__main__":
    main()
