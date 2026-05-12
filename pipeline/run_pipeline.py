# pipeline/run_pipeline.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion     import run as ingest
from src.data_preprocessing import run as preprocess
from src.model_trainer      import train_and_log
from src.monitor            import run_monitoring
from src.logger             import get_logger

logger = get_logger("pipeline")

def run_pipeline():
    logger.info("========================================")
    logger.info("   LANDSLIDE MLOPS PIPELINE STARTED    ")
    logger.info("========================================")

    logger.info("STEP 1/4 — Data Ingestion")
    ingest()

    logger.info("STEP 2/4 — Data Preprocessing")
    preprocess()

    logger.info("STEP 3/4 — Model Training + MLflow")
    train_and_log()

    logger.info("STEP 4/4 — Monitoring Reports")
    run_monitoring()

    logger.info("========================================")
    logger.info("   PIPELINE COMPLETED SUCCESSFULLY      ")
    logger.info("========================================")

if __name__ == "__main__":
    run_pipeline()