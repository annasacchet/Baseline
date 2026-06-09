"""SQuAD-style Answer F1 utilities, shared by the OLMo-32B eval scripts."""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def get_tokens(s: str):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def best_f1(pred: str, golds):
    best_score, best_ref = 0.0, (golds[0] if golds else "")
    for g in golds:
        s = compute_f1(g, pred)
        if s > best_score:
            best_score, best_ref = s, g
    return best_score, best_ref
