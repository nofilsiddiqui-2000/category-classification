"""Rule-based classifier used for seed labeling and fallback predictions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable

from .constants import SUPER_CATEGORIES


def _normalize_text(value: str) -> str:
    text = value.lower().strip()
    text = re.sub(r"[^a-z0-9\s&/+.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_pattern(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword.lower())
    escaped = escaped.replace(r"\ ", r"\s+")
    plural_suffix = ""
    if not keyword.lower().endswith("s"):
        plural_suffix = r"(?:es|s)?"
    return re.compile(rf"(?<![a-z0-9]){escaped}{plural_suffix}(?![a-z0-9])")


@dataclass(frozen=True)
class RuleResult:
    label: str
    confidence: float
    matched_terms: str
    scores: Dict[str, float]


RULE_KEYWORDS = {
    "Food": [
        "bread",
        "bagel",
        "biscuit",
        "crisps",
        "extruded snack",
        "toast",
        "snack",
        "granola",
        "macaroon",
        "cereal",
        "meal",
        "dish",
        "turkey",
        "fruit",
        "veggie",
        "vegetable",
        "frozen vegetable",
        "dairy",
        "milk",
        "butter",
        "cream",
        "yoghurt",
        "yogurt",
        "tuna",
        "chicken",
        "meat",
        "pate",
        "hamburguers",
        "burger",
        "chocolate",
        "sugar",
        "ice cream",
        "salad dressing",
        "olive oil",
        "empanada",
        "candy",
        "eggs",
        "ice cream",
    ],
    "Drinks": [
        "drink",
        "beverage",
        "juice",
        "veggie juice",
        "tea",
        "coffee",
        "coffe",
        "water",
        "soda",
        "cola",
        "alcohol",
        "beer",
        "wine",
        "distilled",
        "rtd",
        "hot chocolate",
        "sport drink",
        "mate cocido",
    ],
    "Home Care": [
        "detergent",
        "bleach",
        "cleaner",
        "cleaners",
        "cleaning",
        "disinfect",
        "disinfectant",
        "drain",
        "declog",
        "laundry",
        "additive",
        "floor",
        "scrubber",
        "steel scrubber",
        "sponge",
        "steel wool",
        "storage bag",
        "plastic storage bag",
        "plastic bag",
        "garbage bag",
        "candle",
        "scented candle",
        "insect repellent",
        "repellent",
        "insecticide",
        "batteries",
        "gloves",
    ],
    "Personal Care": [
        "soap",
        "toothpaste",
        "deodorant",
        "shampoo",
        "conditioner",
        "hair",
        "face care",
        "face",
        "skin care",
        "nail",
        "treatment",
        "shaving",
        "epilator",
        "incontinence",
        "body wash",
    ],
}

NEGATIVE_TERMS = {
    "TOTAL MARKET",
    "TOTAL",
    "MISC",
    "OTHERS",
}


class RuleClassifier:
    """Keyword-score classifier with confidence estimation."""

    def __init__(self, keywords: Dict[str, Iterable[str]] | None = None) -> None:
        self.keywords = keywords or RULE_KEYWORDS
        self.patterns: Dict[str, list[tuple[str, re.Pattern[str], float]]] = {}
        for label in SUPER_CATEGORIES:
            if label == "Other":
                continue
            label_keywords = list(self.keywords.get(label, []))
            compiled: list[tuple[str, re.Pattern[str], float]] = []
            for term in label_keywords:
                weight = 2.0 if " " in term else 1.0
                compiled.append((term, _build_pattern(term), weight))
            self.patterns[label] = compiled

    def predict_one(self, category_name: str) -> RuleResult:
        raw = str(category_name).strip()
        if not raw:
            return RuleResult(
                label="Other",
                confidence=0.0,
                matched_terms="",
                scores={k: 0.0 for k in SUPER_CATEGORIES},
            )

        upper = raw.upper()
        if upper in NEGATIVE_TERMS:
            return RuleResult(
                label="Other",
                confidence=0.95,
                matched_terms="negative_term",
                scores={k: 0.0 for k in SUPER_CATEGORIES},
            )

        text = _normalize_text(raw)
        scores = {k: 0.0 for k in SUPER_CATEGORIES}
        hits: list[str] = []

        for label, pattern_entries in self.patterns.items():
            for term, pattern, weight in pattern_entries:
                if pattern.search(text):
                    scores[label] += weight
                    hits.append(f"{label}:{term}")

        # Light preference for "drink" terms when present.
        if "drink" in text or "beverage" in text:
            scores["Drinks"] += 0.6

        if "rtd" in text:
            scores["Drinks"] += 0.8

        if any(x in text for x in ["juice", "tea", "coffee", "coffe", "hot chocolate"]):
            scores["Drinks"] += 0.55

        # Special-case ambiguity between dairy food and drink terms.
        if "milk" in text and ("drink" in text or "beverage" in text):
            scores["Drinks"] += 0.8

        if any(
            x in text
            for x in [
                "hair",
                "nail",
                "face",
                "skin",
                "incontinence",
                "shav",
                "deodor",
            ]
        ):
            scores["Personal Care"] += 0.7

        if "pet" in text:
            scores["Food"] *= 0.5

        non_other_labels = [l for l in SUPER_CATEGORIES if l != "Other"]
        ranked = sorted(non_other_labels, key=lambda x: scores[x], reverse=True)
        best = ranked[0]
        second = ranked[1]
        best_score = scores[best]
        second_score = scores[second]

        if best_score <= 0:
            return RuleResult(
                label="Other",
                confidence=0.2,
                matched_terms="",
                scores=scores,
            )

        margin = best_score - second_score
        confidence = 0.45 + min(0.3, best_score / 8.0) + min(0.24, margin / 5.0)
        confidence = max(0.0, min(0.99, confidence))

        # Lower confidence for generic short category names.
        if len(text.split()) == 1 and best_score <= 1.0:
            confidence = min(confidence, 0.62)

        return RuleResult(
            label=best,
            confidence=round(confidence, 4),
            matched_terms="; ".join(hits),
            scores=scores,
        )
