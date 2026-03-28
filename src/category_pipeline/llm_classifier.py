"""Optional LLM-based classifier for ambiguous categories."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable

from openai import OpenAI

from .constants import CATEGORY_DEFINITIONS, LABEL_NORMALIZATION


@dataclass(frozen=True)
class LLMResult:
    label: str
    confidence: float
    rationale: str


def _safe_float(value: object, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_json_block(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def _normalize_label(label: str) -> str:
    key = str(label).strip().lower()
    return LABEL_NORMALIZATION.get(key, "Other")


def _build_prompt(categories: Iterable[str]) -> str:
    lines = []
    for idx, category in enumerate(categories, start=1):
        lines.append(f"{idx}. {category}")
    definitions_text = "\n".join(
        f"- {label}: {desc}" for label, desc in CATEGORY_DEFINITIONS.items()
    )
    categories_text = "\n".join(lines)

    return (
        "Classify each category into one label from this set only:\n"
        "Food, Drinks, Home Care, Personal Care, Other.\n\n"
        "Definitions:\n"
        f"{definitions_text}\n\n"
        "Output strict JSON with this schema only:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "index": <int>,\n'
        '      "label": "<Food|Drinks|Home Care|Personal Care|Other>",\n'
        '      "confidence": <float 0-1>,\n'
        '      "rationale": "<short reason>"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Categories:\n"
        f"{categories_text}"
    )


def classify_categories_with_llm(
    categories: list[str],
    model: str = "gpt-4.1-mini",
    batch_size: int = 30,
    max_retries: int = 3,
    api_key: str | None = None,
) -> Dict[str, LLMResult]:
    if not categories:
        return {}
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Cannot run LLM classification.")

    client = OpenAI(api_key=key)
    final_results: Dict[str, LLMResult] = {}

    for offset in range(0, len(categories), batch_size):
        batch = categories[offset : offset + batch_size]
        prompt = _build_prompt(batch)
        parsed = None

        for attempt in range(1, max_retries + 1):
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": "You are a strict JSON classification engine.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            try:
                parsed = _extract_json_block(response.output_text)
                break
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(1.5 * attempt)

        if not parsed or "items" not in parsed:
            raise RuntimeError("Invalid LLM output format.")

        for item in parsed["items"]:
            idx = int(item.get("index", -1))
            if idx < 1 or idx > len(batch):
                continue
            name = batch[idx - 1]
            label = _normalize_label(item.get("label", "Other"))
            confidence = max(0.0, min(1.0, _safe_float(item.get("confidence"), 0.5)))
            rationale = str(item.get("rationale", "")).strip()
            final_results[name] = LLMResult(
                label=label,
                confidence=round(confidence, 4),
                rationale=rationale,
            )

    return final_results
