import re
from typing import Dict

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def exact_match(pred: str, ref: str) -> float:
    """1.0 if normalized strings are identical, else 0.0"""
    return float(_normalize(pred) == _normalize(ref))

def f1_token_overlap(pred: str, ref: str) -> float:
    """
    Simple token-set F1:
    - precision = overlap / unique tokens in prediction
    - recall    = overlap / unique tokens in reference
    """
    pred_tokens = _normalize(pred).split()
    ref_tokens = _normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(ref_tokens))
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

def score_example(pred: str, ref: str) -> Dict[str, float]:
    """Return a small dict of metric scores for one example."""
    return {
        "exact_match": exact_match(pred, ref),
        "f1": f1_token_overlap(pred, ref),
    }
