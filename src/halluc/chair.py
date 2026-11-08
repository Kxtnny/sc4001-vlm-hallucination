"""CHAIR: object hallucination in free-form captions (Rohrbach et al., EMNLP 2018).

We re-implement the scoring logic of the authors' chair.py (github.com/LisaAnne/Hallucination)
and use their synonym list unchanged (downloaded by scripts/prepare_data.py to
data/chair/synonyms.txt). Each of its 80 lines is one COCO category; the first
word is the category ("node") name, the rest are synonyms.

How a caption is scored:
  1. Lower-case it and split it into words (with the character span of each word).
  2. Singularise each word ("dogs" -> "dog"). The original used pattern.en, which
     no longer installs on modern Python; we use `inflect`, and only accept the
     singular form if it is a known object word, so non-object words are never
     turned into objects by mistake.
  3. Join known two-word objects ("hot dog", "teddy bear", "baby elephant" -> elephant ...)
     exactly as in the original code, drop "seat" if "toilet" is also mentioned,
     and map every remaining object word to its COCO category with the synonym list.
  4. Ground truth for an image = categories in the COCO instance annotations plus
     objects found (with the same parser) in its 5 reference captions.
  5. A mentioned category that is not in the ground truth is a hallucination.

Metrics (over a set of captions):
  CHAIR-s = captions with >= 1 hallucinated object / captions
  CHAIR-i = hallucinated object mentions / all object mentions (every mention counts,
            as in the original code)
  recall  = distinct ground-truth categories mentioned / distinct ground-truth categories
            (micro-averaged over images; the usual companion metric, e.g. in OPERA)
  length  = mean caption length in words
"""
import re
from collections import Counter
from pathlib import Path

# Two-word objects and special cases, copied from the original chair.py.
COCO_DOUBLE_WORDS = ["motor bike", "motor cycle", "air plane", "traffic light", "street light",
                     "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
                     "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
                     "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear", "hair drier",
                     "potted plant", "bow tie", "laptop computer", "stove top oven", "home plate",
                     "train track"]
ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "animal", "cub"]
VEHICLE_WORDS = ["jet", "train"]

_WORD_RE = re.compile(r"[a-z]+")


def _default_singularize():
    import inflect
    engine = inflect.engine()

    def singular(word):
        s = engine.singular_noun(word)
        return s if s else word
    return singular


class Chair:
    def __init__(self, synonyms_path, singularize=None, expected_categories=80):
        lines = [ln for ln in Path(synonyms_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        # Strip each entry: the original file has a few double spaces after commas.
        groups = [[w.strip().lower() for w in ln.split(",") if w.strip()] for ln in lines]
        if expected_categories is not None:
            assert len(groups) == expected_categories, (
                f"expected {expected_categories} COCO categories in {synonyms_path}, got {len(groups)}")
        self.inverse_synonym = {}
        for g in groups:
            for w in g:
                self.inverse_synonym[w] = g[0]
        self.categories = [g[0] for g in groups]

        self.double_words = {w: w for w in COCO_DOUBLE_WORDS}
        for a in ANIMAL_WORDS:
            self.double_words[f"baby {a}"] = a
            self.double_words[f"adult {a}"] = a
        for v in VEHICLE_WORDS:
            self.double_words[f"passenger {v}"] = v
        self.double_words["bow tie"] = "tie"
        self.double_words["toilet seat"] = "toilet"

        # Every single word that can be part of an object mention; used to decide
        # whether a singularised form should be accepted.
        self.known_words = set(self.inverse_synonym) | {p for k in self.double_words for p in k.split()}
        self._singular = singularize or _default_singularize()

    def _norm(self, word):
        if word in self.known_words:
            return word
        s = self._singular(word)
        return s if s in self.known_words else word

    def parse(self, caption):
        """Object mentions in a caption, in order.

        Returns a list of dicts: {"word": surface form (after singularising/joining),
        "node": COCO category, "start": char offset, "end": char offset}.
        """
        text = caption.lower()
        toks = [(self._norm(m.group()), m.start(), m.end()) for m in _WORD_RE.finditer(text)]
        merged, i = [], 0
        while i < len(toks):
            pair = " ".join(t[0] for t in toks[i:i + 2])
            if i + 1 < len(toks) and pair in self.double_words:
                merged.append((self.double_words[pair], toks[i][1], toks[i + 1][2]))
                i += 2
            else:
                merged.append(toks[i])
                i += 1
        words = [w for w, _, _ in merged]
        if "toilet" in words and "seat" in words:  # "the seat of the toilet" is not a chair
            merged = [t for t in merged if t[0] != "seat"]
        return [{"word": w, "node": self.inverse_synonym[w], "start": s, "end": e}
                for w, s, e in merged if w in self.inverse_synonym]

    def ground_truth(self, instance_category_names, reference_captions):
        gt = {self.inverse_synonym[n.lower()] for n in instance_category_names}
        for c in reference_captions:
            gt |= {m["node"] for m in self.parse(c)}
        return gt

    def score_caption(self, caption, gt):
        mentions = self.parse(caption)
        for m in mentions:
            m["hallucinated"] = m["node"] not in gt
        mentioned = {m["node"] for m in mentions}
        return {
            "mentions": mentions,
            "n_mentions": len(mentions),
            "n_hallucinated": sum(m["hallucinated"] for m in mentions),
            "hallucinated_nodes": sorted({m["node"] for m in mentions if m["hallucinated"]}),
            "correct_nodes": sorted(mentioned & gt),
            "gt_nodes": sorted(gt),
            "n_words": len(_WORD_RE.findall(caption.lower())),
        }


def chair_metrics(scored):
    """scored: list of score_caption() outputs (one per image)."""
    n = len(scored)
    mentions = sum(s["n_mentions"] for s in scored)
    halluc = sum(s["n_hallucinated"] for s in scored)
    gt_total = sum(len(s["gt_nodes"]) for s in scored)
    covered = sum(len(s["correct_nodes"]) for s in scored)
    return {
        "n_captions": n,
        "chair_s": sum(s["n_hallucinated"] > 0 for s in scored) / n if n else 0.0,
        "chair_i": halluc / mentions if mentions else 0.0,
        "recall": covered / gt_total if gt_total else 0.0,
        "avg_len_words": sum(s["n_words"] for s in scored) / n if n else 0.0,
        "avg_mentions": mentions / n if n else 0.0,
        "n_mentions": mentions,
        "n_hallucinated_mentions": halluc,
    }


def coco_ground_truth_tables(instances, captions):
    """Per-image COCO category names and reference captions, keyed by image file name."""
    from .data import coco_file_name
    cat_name = {c["id"]: c["name"] for c in instances["categories"]}
    cats, caps = {}, {}
    for a in instances["annotations"]:
        cats.setdefault(coco_file_name(a["image_id"]), set()).add(cat_name[a["category_id"]])
    for a in captions["annotations"]:
        caps.setdefault(coco_file_name(a["image_id"]), []).append(a["caption"])
    return cats, caps


def hallucination_counts(scored):
    """Which categories are hallucinated most often (for the failure analysis)."""
    return Counter(n for s in scored for n in s["hallucinated_nodes"])
