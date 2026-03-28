"""ML model utilities for pseudo-label self-training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class TrainOutput:
    model: Pipeline
    metrics: Dict[str, float]


def _build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=None,
                ),
            ),
        ]
    )


def train_text_classifier(
    texts: Sequence[str],
    labels: Sequence[str],
    random_state: int = 42,
) -> TrainOutput | None:
    if len(texts) < 12:
        return None
    distinct = set(labels)
    if len(distinct) < 2:
        return None

    model = _build_model(random_state=random_state)
    label_counts = {label: labels.count(label) for label in distinct}
    can_stratify = all(count >= 2 for count in label_counts.values()) and len(texts) >= 20

    if can_stratify:
        x_train, x_val, y_train, y_val = train_test_split(
            list(texts),
            list(labels),
            test_size=0.2,
            random_state=random_state,
            stratify=list(labels),
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        metrics = {
            "val_accuracy": float(accuracy_score(y_val, y_pred)),
            "val_f1_macro": float(f1_score(y_val, y_pred, average="macro")),
            "train_samples": float(len(x_train)),
            "val_samples": float(len(x_val)),
        }
    else:
        model.fit(list(texts), list(labels))
        metrics = {
            "val_accuracy": -1.0,
            "val_f1_macro": -1.0,
            "train_samples": float(len(texts)),
            "val_samples": 0.0,
        }

    return TrainOutput(model=model, metrics=metrics)


def predict_with_confidence(model: Pipeline, texts: Iterable[str]) -> tuple[list[str], np.ndarray]:
    text_list = list(texts)
    labels = model.predict(text_list).tolist()
    probs = model.predict_proba(text_list)
    max_probs = probs.max(axis=1)
    return labels, max_probs
