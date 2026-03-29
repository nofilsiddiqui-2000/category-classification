"""Semantic prototype scorer for zero-label category classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .constants import CATEGORY_DEFINITIONS
from .rule_engine import RULE_KEYWORDS


@dataclass(frozen=True)
class SemanticResult:
    label: str
    confidence: float
    max_similarity: float
    scores: Dict[str, float]


class PrototypeSemanticScorer:
    """Scores category names against super-category prototype texts."""

    def __init__(self) -> None:
        self.labels = ["Food", "Drinks", "Home Care", "Personal Care"]
        self.prototype_texts = self._build_prototype_texts()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
        self.prototype_matrix = None

    def _build_prototype_texts(self) -> Dict[str, str]:
        texts: Dict[str, str] = {}
        for label in self.labels:
            definition = CATEGORY_DEFINITIONS.get(label, "")
            keywords = " ".join(RULE_KEYWORDS.get(label, []))
            texts[label] = f"{definition} {keywords}".strip()
        return texts

    def fit(self, category_names: Iterable[str]) -> None:
        category_texts = [str(x).strip() for x in category_names if str(x).strip()]
        corpus = list(self.prototype_texts.values()) + category_texts
        self.vectorizer.fit(corpus)
        self.prototype_matrix = self.vectorizer.transform(
            [self.prototype_texts[label] for label in self.labels]
        )

    def predict_one(self, category_name: str) -> SemanticResult:
        if self.prototype_matrix is None:
            raise RuntimeError("PrototypeSemanticScorer is not fitted.")

        text = str(category_name).strip()
        if not text:
            return SemanticResult(
                label="Other",
                confidence=0.0,
                max_similarity=0.0,
                scores={label: 0.0 for label in self.labels},
            )

        vector = self.vectorizer.transform([text])
        similarities = cosine_similarity(vector, self.prototype_matrix)[0]
        score_map = {label: float(similarities[i]) for i, label in enumerate(self.labels)}

        ranked = sorted(self.labels, key=lambda x: score_map[x], reverse=True)
        best_label = ranked[0]
        second_label = ranked[1]
        best_sim = score_map[best_label]
        second_sim = score_map[second_label]
        margin = best_sim - second_sim

        # Map similarity + margin to confidence.
        confidence = (best_sim * 0.85) + (max(0.0, margin) * 0.65)
        confidence = max(0.0, min(0.99, confidence))

        # Similarity too low means it is likely out of taxonomy.
        if best_sim < 0.11:
            return SemanticResult(
                label="Other",
                confidence=round(min(0.55, 0.2 + best_sim), 4),
                max_similarity=round(best_sim, 4),
                scores=score_map,
            )

        return SemanticResult(
            label=best_label,
            confidence=round(confidence, 4),
            max_similarity=round(best_sim, 4),
            scores=score_map,
        )

    def predict_many(self, category_names: Iterable[str]) -> Dict[str, SemanticResult]:
        names = [str(x).strip() for x in category_names]
        self.fit(names)
        return {name: self.predict_one(name) for name in names}
