"""
Training Runner — P4 Radiology AI

Run with: uv run python scripts/run_training.py

Requires: full NIH dataset on disk (~2-4 hours on T4 GPU)
"""

import logging

import yaml
import pandas as pd

from src.data_prep import prepare_dataset
from src.train import train_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

CONFIG_PATH = "config/training_config.yaml"

config = yaml.safe_load(open(CONFIG_PATH))

print("Preparing dataset splits...")
train_df, val_df, test_df, split_hash = prepare_dataset(
    labels_csv_path=config["data"]["labels_path"],
    images_dir=config["data"]["dataset_path"],
    config=config,
)

print(f"Split hash: {split_hash}")
print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# Save val and test splits for L7 evaluation
test_df.to_parquet("artifacts/test_df.parquet", index=False)
val_df.to_parquet("artifacts/val_df.parquet", index=False)

print("Splits saved to artifacts/ for L7 evaluation.")

print("\nStarting two-phase training...")
model, run_id = train_pipeline(CONFIG_PATH, train_df, val_df)

print(f"\nTraining complete.")
print(f"MLflow run_id: {run_id}")
print(f"View results:  uv run mlflow ui")
print(f"Best model:    artifacts/best_model.pt")