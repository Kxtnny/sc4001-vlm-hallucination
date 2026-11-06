"""Yes/no answer parsing and POPE metrics.

How answers are parsed (state this in the report's Methods section):
  1. Lower-case the answer and split it into words.
  2. If the first word is "yes" or "no", use it.            (normal case)
  3. Otherwise fall back to the rule in the original POPE evaluation code:
     answer "no" if the words contain "no" or "not", else "yes".
Step 3 is marked with parse_fallback=True so we can count how often it is used.
If that count is more than a handful, read those answers by hand.
"""
import re


def parse_yes_no(text):
    """Return (prediction, used_fallback) where prediction is "yes" or "no"."""
    words = re.findall(r"[a-z]+", text.lower())
    if words and words[0] in ("yes", "no"):
        return words[0], False
    if "no" in words or "not" in words:
        return "no", True
    return "yes", True


def pope_metrics(records):
    """records: dicts with "pred" and "label" ("yes"/"no"); "yes" is the positive class.

    yes_ratio is the share of "yes" predictions. POPE is balanced (50% yes), so a
    yes_ratio well above 0.5 means the model tends to say an object is there
    when it is not, which is the hallucination we care about.
    """
    tp = sum(r["pred"] == "yes" and r["label"] == "yes" for r in records)
    fp = sum(r["pred"] == "yes" and r["label"] == "no" for r in records)
    tn = sum(r["pred"] == "no" and r["label"] == "no" for r in records)
    fn = sum(r["pred"] == "no" and r["label"] == "yes" for r in records)
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": (tp + fp) / n if n else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_parse_fallback": sum(bool(r.get("parse_fallback")) for r in records),
    }
