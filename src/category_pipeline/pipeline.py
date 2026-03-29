"""Task 1 end-to-end category classification pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from .evaluation import evaluate_predictions
from .io_utils import infer_category_column, read_table, write_table
from .llm_classifier import classify_categories_with_llm
from .modeling import predict_with_confidence, train_text_classifier
from .rule_engine import RuleClassifier
from .semantic_scorer import PrototypeSemanticScorer


def _to_key(value: Any) -> str:
    return str(value).strip()


@dataclass
class PipelineConfig:
    llm_trigger_confidence: float = 0.62
    model_seed_min_confidence: float = 0.58
    semantic_seed_min_confidence: float = 0.56
    semantic_override_margin: float = 0.08
    rule_final_confidence: float = 0.8
    llm_final_confidence: float = 0.68
    model_final_confidence: float = 0.57
    semantic_final_confidence: float = 0.6
    random_state: int = 42
    llm_batch_size: int = 30
    llm_model: str = "gpt-4.1-mini"

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "PipelineConfig":
        if not path:
            return cls()
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("Config YAML must be a key/value object.")
        return cls(**payload)


@dataclass
class PipelineSummary:
    input_path: str
    output_path: str
    artifacts_dir: str
    category_column: str
    input_rows: int
    unique_categories: int
    used_llm: bool
    trained_ml_model: bool
    source_counts: dict[str, int]
    label_counts: dict[str, int]
    model_metrics: dict[str, float]
    eval_metrics: dict[str, Any]
    target_accuracy: float | None
    target_met: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    category_column: str | None = None,
    artifacts_dir: str | Path = "artifacts/task1",
    enable_llm: bool = False,
    config_path: str | Path | None = None,
    llm_model: str | None = None,
    llm_batch_size: int | None = None,
    llm_api_key: str | None = None,
    label_column: str | None = None,
    target_accuracy: float | None = None,
    enforce_target: bool = False,
) -> PipelineSummary:
    cfg = PipelineConfig.from_yaml(config_path)
    if llm_model:
        cfg.llm_model = llm_model
    if llm_batch_size:
        cfg.llm_batch_size = llm_batch_size

    input_path = Path(input_path)
    output_path = Path(output_path)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path)
    category_col = infer_category_column(df, preferred=category_column)
    working_df = df.copy()
    working_df["_category_key"] = working_df[category_col].map(_to_key)

    output_cols = [
        "super_category",
        "super_category_confidence",
        "super_category_source",
    ]
    effective_label_col = label_column
    if label_column and label_column in output_cols and label_column in working_df.columns:
        backup_col = "__true_label_backup__"
        working_df[backup_col] = working_df[label_column]
        effective_label_col = backup_col

    # Avoid merge suffix collisions when input already has prediction columns.
    drop_collision_cols = [c for c in output_cols if c in working_df.columns]
    if drop_collision_cols:
        working_df = working_df.drop(columns=drop_collision_cols)

    unique_categories = sorted(
        [x for x in working_df["_category_key"].dropna().unique().tolist() if x]
    )

    unique_df = pd.DataFrame({"_category_key": unique_categories})
    rule_model = RuleClassifier()
    rule_results = unique_df["_category_key"].map(rule_model.predict_one)
    unique_df["rule_label"] = rule_results.map(lambda x: x.label)
    unique_df["rule_confidence"] = rule_results.map(lambda x: x.confidence)
    unique_df["rule_terms"] = rule_results.map(lambda x: x.matched_terms)

    semantic_model = PrototypeSemanticScorer()
    semantic_results = semantic_model.predict_many(unique_df["_category_key"].tolist())
    unique_df["semantic_label"] = unique_df["_category_key"].map(
        lambda x: semantic_results[x].label
    )
    unique_df["semantic_confidence"] = unique_df["_category_key"].map(
        lambda x: semantic_results[x].confidence
    )
    unique_df["semantic_similarity"] = unique_df["_category_key"].map(
        lambda x: semantic_results[x].max_similarity
    )

    unique_df["llm_label"] = pd.Series([pd.NA] * len(unique_df), dtype="object")
    unique_df["llm_confidence"] = pd.Series([float("nan")] * len(unique_df), dtype="float")
    unique_df["llm_rationale"] = pd.Series([pd.NA] * len(unique_df), dtype="object")
    used_llm = False

    if enable_llm:
        llm_mask = (unique_df["rule_confidence"] < cfg.llm_trigger_confidence) | (
            unique_df["rule_label"] == "Other"
        )
        llm_candidates = unique_df.loc[llm_mask, "_category_key"].tolist()
        if llm_candidates:
            llm_results = classify_categories_with_llm(
                categories=llm_candidates,
                model=cfg.llm_model,
                batch_size=cfg.llm_batch_size,
                api_key=llm_api_key,
            )
            used_llm = True
            if llm_results:
                unique_df["llm_label"] = unique_df["_category_key"].map(
                    lambda x: llm_results[x].label if x in llm_results else pd.NA
                )
                unique_df["llm_confidence"] = unique_df["_category_key"].map(
                    lambda x: llm_results[x].confidence if x in llm_results else pd.NA
                )
                unique_df["llm_rationale"] = unique_df["_category_key"].map(
                    lambda x: llm_results[x].rationale if x in llm_results else pd.NA
                )

    # Seed labels for self-training.
    unique_df["seed_label"] = unique_df["rule_label"]
    unique_df["seed_confidence"] = unique_df["rule_confidence"]

    # If semantic and rule agree, boost confidence.
    agree_mask = (
        (unique_df["semantic_label"] == unique_df["rule_label"])
        & (unique_df["semantic_label"] != "Other")
    )
    if agree_mask.any():
        unique_df.loc[agree_mask, "seed_confidence"] = unique_df.loc[
            agree_mask, ["rule_confidence", "semantic_confidence"]
        ].max(axis=1)

    # Semantic override for rule misses/weak rule predictions.
    semantic_override_mask = (
        (unique_df["semantic_label"] != "Other")
        & (
            (unique_df["rule_label"] == "Other")
            | (
                unique_df["semantic_confidence"]
                >= unique_df["rule_confidence"] + cfg.semantic_override_margin
            )
        )
        & (unique_df["semantic_confidence"] >= cfg.semantic_seed_min_confidence)
    )
    if semantic_override_mask.any():
        unique_df.loc[semantic_override_mask, "seed_label"] = unique_df.loc[
            semantic_override_mask, "semantic_label"
        ]
        unique_df.loc[semantic_override_mask, "seed_confidence"] = unique_df.loc[
            semantic_override_mask, "semantic_confidence"
        ]

    llm_available_mask = unique_df["llm_label"].notna()
    if llm_available_mask.any():
        unique_df.loc[llm_available_mask, "seed_label"] = unique_df.loc[
            llm_available_mask, "llm_label"
        ]
        unique_df.loc[llm_available_mask, "seed_confidence"] = unique_df.loc[
            llm_available_mask, "llm_confidence"
        ]

    train_mask = unique_df["seed_confidence"] >= cfg.model_seed_min_confidence
    seed_df = unique_df.loc[train_mask, ["_category_key", "seed_label"]].copy()
    train_output = None
    if len(seed_df) > 0:
        train_output = train_text_classifier(
            seed_df["_category_key"].tolist(),
            seed_df["seed_label"].tolist(),
            random_state=cfg.random_state,
        )

    trained_ml_model = train_output is not None
    unique_df["ml_label"] = pd.NA
    unique_df["ml_confidence"] = pd.NA
    model_metrics: dict[str, float] = {}
    if trained_ml_model and train_output is not None:
        ml_labels, ml_probs = predict_with_confidence(
            train_output.model, unique_df["_category_key"].tolist()
        )
        unique_df["ml_label"] = ml_labels
        unique_df["ml_confidence"] = ml_probs.round(4)
        model_metrics = train_output.metrics
        joblib.dump(train_output.model, artifacts_dir / "task1_text_classifier.joblib")

    # Final decision logic.
    final_labels: list[str] = []
    final_confidences: list[float] = []
    final_sources: list[str] = []

    for row in unique_df.itertuples(index=False):
        llm_label = getattr(row, "llm_label")
        llm_conf = getattr(row, "llm_confidence")
        rule_label = getattr(row, "rule_label")
        rule_conf = float(getattr(row, "rule_confidence"))
        semantic_label = getattr(row, "semantic_label")
        semantic_conf = float(getattr(row, "semantic_confidence"))
        ml_label = getattr(row, "ml_label")
        ml_conf = getattr(row, "ml_confidence")

        if pd.notna(llm_label) and float(llm_conf) >= cfg.llm_final_confidence:
            final_labels.append(str(llm_label))
            final_confidences.append(float(llm_conf))
            final_sources.append("llm")
            continue

        if rule_label == "Other" and rule_conf >= 0.9:
            final_labels.append("Other")
            final_confidences.append(rule_conf)
            final_sources.append("rule")
            continue

        if (
            semantic_label == rule_label
            and semantic_label != "Other"
            and max(semantic_conf, rule_conf)
            >= min(cfg.rule_final_confidence, cfg.semantic_final_confidence)
        ):
            final_labels.append(str(semantic_label))
            final_confidences.append(max(semantic_conf, rule_conf))
            final_sources.append("rule_semantic")
            continue

        if (
            semantic_label != "Other"
            and semantic_conf >= cfg.semantic_final_confidence
            and semantic_conf >= rule_conf - 0.05
        ):
            final_labels.append(str(semantic_label))
            final_confidences.append(semantic_conf)
            final_sources.append("semantic")
            continue

        if rule_label != "Other" and rule_conf >= cfg.rule_final_confidence:
            final_labels.append(str(rule_label))
            final_confidences.append(rule_conf)
            final_sources.append("rule")
            continue

        if pd.notna(ml_label) and float(ml_conf) >= cfg.model_final_confidence:
            final_labels.append(str(ml_label))
            final_confidences.append(float(ml_conf))
            final_sources.append("ml")
            continue

        if rule_label != "Other" and rule_conf >= 0.45:
            final_labels.append(str(rule_label))
            final_confidences.append(rule_conf)
            final_sources.append("rule_low_conf")
            continue

        final_labels.append("Other")
        final_confidences.append(max(0.35, rule_conf))
        final_sources.append("fallback_other")

    unique_df["super_category"] = final_labels
    unique_df["super_category_confidence"] = [round(x, 4) for x in final_confidences]
    unique_df["super_category_source"] = final_sources

    result_df = working_df.merge(
        unique_df[
            [
                "_category_key",
                "super_category",
                "super_category_confidence",
                "super_category_source",
            ]
        ],
        on="_category_key",
        how="left",
    ).drop(columns=["_category_key"])

    write_table(result_df, output_path)
    unique_df.sort_values("_category_key").to_csv(
        artifacts_dir / "unique_category_predictions.csv", index=False
    )
    (artifacts_dir / "pipeline_config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    if model_metrics:
        (artifacts_dir / "model_metrics.json").write_text(
            json.dumps(model_metrics, indent=2), encoding="utf-8"
        )

    eval_metrics: dict[str, Any] = {}
    target_met: bool | None = None
    if effective_label_col:
        eval_result = evaluate_predictions(
            result_df, true_label_col=effective_label_col, pred_label_col="super_category"
        )
        eval_metrics = {
            "accuracy": eval_result.accuracy,
            "f1_macro": eval_result.f1_macro,
            "samples": eval_result.samples,
            "per_label_support": eval_result.per_label_support,
        }
        (artifacts_dir / "eval_metrics.json").write_text(
            json.dumps(eval_metrics, indent=2), encoding="utf-8"
        )
        if target_accuracy is not None:
            target_met = eval_result.accuracy >= float(target_accuracy)
            if enforce_target and not target_met:
                raise RuntimeError(
                    "Target accuracy not met: "
                    f"{eval_result.accuracy:.4f} < {float(target_accuracy):.4f}"
                )

    summary = PipelineSummary(
        input_path=str(input_path),
        output_path=str(output_path),
        artifacts_dir=str(artifacts_dir),
        category_column=category_col,
        input_rows=len(df),
        unique_categories=len(unique_df),
        used_llm=used_llm,
        trained_ml_model=trained_ml_model,
        source_counts=result_df["super_category_source"].value_counts().to_dict(),
        label_counts=result_df["super_category"].value_counts().to_dict(),
        model_metrics=model_metrics,
        eval_metrics=eval_metrics,
        target_accuracy=target_accuracy,
        target_met=target_met,
    )
    (artifacts_dir / "run_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    return summary
