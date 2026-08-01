"""
Explainability Runner — P4 Radiology AI

Run with: uv run python scripts/run_explainability.py

Requires: L8 complete (fp_cases.parquet, fn_cases.parquet exist)
"""

import logging
import sys

import pandas as pd

from src.explain import run_explainability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

RUN_ID = input("Enter the MLflow run_id from L6: ").strip()

if not RUN_ID:
    print("ERROR: run_id required.")
    sys.exit(1)

# Load failure cases from L8
fn_df = pd.read_parquet("artifacts/fn_cases.parquet")
fp_df = pd.read_parquet("artifacts/fp_cases.parquet")
all_test_df = pd.read_parquet("artifacts/test_df.parquet")

print(f"FN cases: {len(fn_df):,}  |  FP cases: {len(fp_df):,}")

results = run_explainability(
    config_path="config/training_config.yaml",
    fn_df=fn_df,
    fp_df=fp_df,
    all_test_df=all_test_df,
    run_id=RUN_ID,
)

print("\n" + "=" * 60)
print("EXPLAINABILITY COMPLETE")
print("=" * 60)
print(f"FN heatmaps:  {len(results['fn_heatmaps'])}")
print(f"FP heatmaps:  {len(results['fp_heatmaps'])}")
print(f"TP heatmaps:  {len(results['tp_heatmaps'])}")
print(f"TN heatmaps:  {len(results['tn_heatmaps'])}")
print()
print("Heatmaps saved to: reports/gradcam/")
print("Summary:           reports/gradcam/summary.md")
print()
print("Next steps:")
print("  1. Review heatmaps in reports/gradcam/")
print("  2. Complete 'Clinical Observations' section in summary.md")
print("  3. Commit when complete")