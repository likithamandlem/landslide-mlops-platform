# src/config.py
import os
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR           = ROOT_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH  = MODELS_DIR / "model.pkl"

# Logs & Reports
LOGS_DIR    = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"

# MLflow — file URI format works on Windows
MLFLOW_TRACKING_URI    = (ROOT_DIR / "mlruns").as_uri()
MLFLOW_EXPERIMENT_NAME = "housing-price-prediction"

# Data source
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"

# Model parameters
TEST_SIZE    = 0.2
RANDOM_STATE = 42
TARGET_COL   = "median_house_value"

# Create all directories on import
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)