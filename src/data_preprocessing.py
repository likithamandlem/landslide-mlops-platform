# src/data_preprocessing.py
import pandas as pd
from sklearn.impute import SimpleImputer
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET_COL
from src.logger import get_logger

logger = get_logger("data_preprocessing")

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing start")

    target = df[TARGET_COL].reset_index(drop=True)
    df = df.drop(TARGET_COL, axis=1)

    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    # Fill missing values with column mean
    imputer = SimpleImputer(strategy="mean")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    df[TARGET_COL] = target
    logger.info(f"Preprocessing done — shape: {df.shape}")
    return df

def run():
    logger.info("=== PREPROCESSING START ===")

    train = pd.read_csv(RAW_DATA_DIR / "train.csv")
    test  = pd.read_csv(RAW_DATA_DIR / "test.csv")

    train_processed = preprocess(train)
    test_processed  = preprocess(test)

    # Align columns — test must have same columns as train
    train_cols = [c for c in train_processed.columns if c != TARGET_COL]
    for col in train_cols:
        if col not in test_processed.columns:
            test_processed[col] = 0
    test_processed = test_processed[train_processed.columns]

    train_processed.to_csv(PROCESSED_DATA_DIR / "train_processed.csv", index=False)
    test_processed.to_csv(PROCESSED_DATA_DIR  / "test_processed.csv",  index=False)

    logger.info("=== PREPROCESSING COMPLETE ===")

if __name__ == "__main__":
    run()