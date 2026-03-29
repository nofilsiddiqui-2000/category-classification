"""Evaluation utilities for labeled category datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .constants import LABEL_NORMALIZATION, SUPER_CATEGORIES


def normalize_label(value: object) -> str:
    key = str(value).strip().lower()
    return LABEL_NORMALIZATION.get(key, "Other")


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    f1_macro: float
    samples: int
    per_label_support: Dict[str, int]


def evaluate_predictions(
    df: pd.DataFrame,
    true_label_col: str,
    pred_label_col: str = "super_category",
) -> EvaluationResult:
    if true_label_col not in df.columns:
        raise ValueError(f"Label column '{true_label_col}' not found in dataframe.")
    if pred_label_col not in df.columns:
        raise ValueError(f"Prediction column '{pred_label_col}' not found in dataframe.")

    eval_df = df[[true_label_col, pred_label_col]].copy()
    eval_df = eval_df.dropna(subset=[true_label_col, pred_label_col])
    if eval_df.empty:
        return EvaluationResult(
            accuracy=0.0,
            f1_macro=0.0,
            samples=0,
            per_label_support={label: 0 for label in SUPER_CATEGORIES},
        )

    y_true = eval_df[true_label_col].map(normalize_label).tolist()
    y_pred = eval_df[pred_label_col].map(normalize_label).tolist()

    support: Dict[str, int] = {label: 0 for label in SUPER_CATEGORIES}
    for item in y_true:
        if item in support:
            support[item] += 1
        else:
            support["Other"] += 1

    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro")),
        samples=len(y_true),
        per_label_support=support,
    )
