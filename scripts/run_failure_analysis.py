"""
Failure Analysis Runner — P4 Radiology AI

Run with: uv run python scripts/run_failure_analysis.py
"""

import logging
import sys

import pandas as pd

from src.failure_analysis import run_failure_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

RUN_ID = input("Enter the MLflow run_id from L6: ").strip()

if not RUN_ID:
    print("ERROR: run_id required.")
    sys.exit(1)

test_df = pd.read_parquet("artifacts/test_df.parquet")

print(f"Test set: {len(test_df):,} images")

results = run_failure_analysis(
    config_path="config/training_config.yaml",
    test_df=test_df,
    run_id=RUN_ID,
)

print("\n" + "=" * 60)
print("FAILURE ANALYSIS COMPLETE")
print("=" * 60)
print(f"FP: {results['fp_count']:,}  |  FN: {results['fn_count']:,}")
print(f"FN high-confidence: {results.get('fn_high_conf_count', '?')} (model was certain — wrong)")
print(f"FN low-confidence:  {results.get('fn_low_conf_count', '?')} (boundary cases)")
print(f"FP high-confidence: {results.get('fp_high_conf_count', '?')} (systematic over-prediction)")
print()
print("Demographic recall:")
for k, v in results.items():
    if k.startswith("recall_") and not k.endswith("_ci") and v is not None:
        print(f"  {k}: {v:.4f}")
print()
print("View position:")
for pos in ["AP", "PA"]:
    r = results.get(f"recall_{pos}", "?")
    p = results.get(f"precision_{pos}", "?")
    print(f"  {pos}: recall={r}  precision={p}")
print()
print("Artifacts saved:")
print("  artifacts/fp_cases.parquet")
print("  artifacts/fn_cases.parquet")
print("  reports/failure_report.md")
print()
print("Next steps:")
print("  1. Fill in all [populate] fields in reports/failure_report.md")
print("  2. Send 'Clinical Advisor Review Template' section to domain advisor")
print("  3. Paste advisor responses into the report")
print("  4. Commit when complete")