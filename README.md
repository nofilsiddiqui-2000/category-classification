# Category ML - Task 1 Pipeline (Category Classification)

This repository now contains a complete end-to-end pipeline for **Task 1: classify category names into super categories**:

- `Food`
- `Drinks`
- `Home Care`
- `Personal Care`
- `Other` (fallback when none of the four apply)

## What the pipeline does

1. Reads category file (`.csv` or `.xlsx`).
2. Detects the category text column (or uses `--category-column`).
3. Runs a rule-based classifier for initial labels.
4. Optionally classifies ambiguous cases using OpenAI (`--enable-llm`).
5. Trains a text model on pseudo-labels (self-training).
6. Produces final category labels with confidence and source.
7. Saves:
   - output file with added columns
   - unique-category prediction table
   - model/metrics/config artifacts

## Added output columns

- `super_category`
- `super_category_confidence`
- `super_category_source`

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python scripts/run_task1_pipeline.py \
  --input Guide/Category_Names.xlsx \
  --output outputs/Category_Names_classified.xlsx \
  --config configs/task1_default.yaml
```

### Optional: enable LLM for ambiguous categories

```bash
export OPENAI_API_KEY="YOUR_KEY"
.venv/bin/python scripts/run_task1_pipeline.py \
  --input Guide/Category_Names.xlsx \
  --output outputs/Category_Names_classified_llm.xlsx \
  --enable-llm \
  --llm-model gpt-4.1-mini
```

Or pass key directly:

```bash
.venv/bin/python scripts/run_task1_pipeline.py \
  --input Guide/Category_Names.xlsx \
  --output outputs/Category_Names_classified_llm.xlsx \
  --enable-llm \
  --llm-model gpt-4.1-mini \
  --api-key "YOUR_KEY"
```

## Artifacts

Default artifact directory: `artifacts/task1/`

- `unique_category_predictions.csv`
- `run_summary.json`
- `pipeline_config.json`
- `model_metrics.json` (if ML model trained)
- `task1_text_classifier.joblib` (if ML model trained)
