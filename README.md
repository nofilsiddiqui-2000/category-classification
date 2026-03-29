# Category Classification (Task 1)

This is the simplest way to run the project.

## Windows (Git Bash) - Use These Exact Commands

Run from `Category_ML` folder:

```bash
cd /d/Projects/Category_ML
/c/Windows/py.exe -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/run_task1_pipeline.py --input Guide/Category_Names.xlsx --output outputs/Category_Names_classified.xlsx --config configs/task1_default.yaml
```

Output file:

- `outputs/Category_Names_classified.xlsx`

## Windows (PowerShell) - Same Flow

```powershell
cd D:\Projects\Category_ML
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_task1_pipeline.py --input Guide\Category_Names.xlsx --output outputs\Category_Names_classified.xlsx --config configs\task1_default.yaml
```

## Accuracy Check (Only If You Have True Labels)

Example command:

```bash
.venv/Scripts/python.exe scripts/run_task1_pipeline.py --input data/labeled_categories.xlsx --output outputs/labeled_categories_pred.xlsx --config configs/task1_default.yaml --label-column true_super_category --target-accuracy 0.90 --enforce-target
```

If accuracy is below `0.90`, the command exits with error.

## Quick Troubleshooting

If you see `ModuleNotFoundError: No module named 'joblib'`:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

If `.venv` is broken, recreate it:

```bash
rm -rf .venv
/c/Windows/py.exe -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## What Gets Added To Output

- `super_category`
- `super_category_confidence`
- `super_category_source`
