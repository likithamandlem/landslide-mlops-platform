# src/data_ingestion.py
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_URL, RAW_DATA_DIR, TEST_SIZE, RANDOM_STATE
from src.logger import get_logger

logger = get_logger("data_ingestion")

def load_data() -> pd.DataFrame:
    logger.info(f"Loading data from: {DATA_URL}")
    df = pd.read_csv(DATA_URL)
    logger.info(f"Data loaded — shape: {df.shape}")
    return df

def split_and_save(df: pd.DataFrame):
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    train_path = RAW_DATA_DIR / "train.csv"
    test_path  = RAW_DATA_DIR / "test.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    logger.info(f"Train saved → {train_path}  shape: {train.shape}")
    logger.info(f"Test saved  → {test_path}   shape: {test.shape}")
    return train_path, test_path

def run():
    logger.info("=== DATA INGESTION START ===")
    df = load_data()
    split_and_save(df)
    logger.info("=== DATA INGESTION COMPLETE ===")

if __name__ == "__main__":
    run()