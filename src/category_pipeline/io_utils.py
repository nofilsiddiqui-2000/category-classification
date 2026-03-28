"""Input/output and schema helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .constants import DEFAULT_CATEGORY_COLUMN_CANDIDATES


def read_table(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    raise ValueError(f"Unsupported input format: {suffix}. Use CSV or XLSX.")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
        return
    raise ValueError(f"Unsupported output format: {suffix}. Use CSV or XLSX.")


def infer_category_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred:
        if preferred in df.columns:
            return preferred
        raise ValueError(
            f"Category column '{preferred}' not found. Available columns: {list(df.columns)}"
        )

    for candidate in DEFAULT_CATEGORY_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    object_cols = [c for c in df.columns if df[c].dtype == object]
    if len(object_cols) == 1:
        return object_cols[0]
    if object_cols:
        return object_cols[0]

    raise ValueError(
        "No text-like category column found. Pass --category-column explicitly."
    )
