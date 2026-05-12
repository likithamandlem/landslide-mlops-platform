# src/monitor.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from src.config import PROCESSED_DATA_DIR, REPORTS_DIR, TARGET_COL
from src.logger import get_logger

logger = get_logger("monitor")

def run_monitoring():
    logger.info("=== MONITORING START ===")

    train = pd.read_csv(PROCESSED_DATA_DIR / "train_processed.csv")
    test  = pd.read_csv(PROCESSED_DATA_DIR / "test_processed.csv")

    # Use only feature columns (drop target)
    feature_cols = [c for c in train.columns if c != TARGET_COL]
    ref  = train[feature_cols]
    curr = test[feature_cols]

    # ── Data Drift Report ─────────────────────────────────────
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=ref, current_data=curr)

    drift_path = REPORTS_DIR / "drift_report.html"
    drift_report.save_html(str(drift_path))
    logger.info(f"Drift report saved → {drift_path}")

    # ── Data Quality Report ───────────────────────────────────
    quality_report = Report(metrics=[DataQualityPreset()])
    quality_report.run(reference_data=ref, current_data=curr)

    quality_path = REPORTS_DIR / "quality_report.html"
    quality_report.save_html(str(quality_path))
    logger.info(f"Quality report saved → {quality_path}")

    logger.info("=== MONITORING COMPLETE ===")

if __name__ == "__main__":
    run_monitoring()