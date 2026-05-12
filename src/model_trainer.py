# src/model_trainer.py
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, MODEL_PATH,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    TARGET_COL, RANDOM_STATE
)
from src.logger import get_logger

logger = get_logger("model_trainer")

def load_data():
    train = pd.read_csv(PROCESSED_DATA_DIR / "train_processed.csv")
    test  = pd.read_csv(PROCESSED_DATA_DIR / "test_processed.csv")
    return train, test

def train_and_log():
    logger.info("=== MODEL TRAINING START ===")

    train, test = load_data()

    X_train = train.drop(TARGET_COL, axis=1)
    y_train = train[TARGET_COL]
    X_test  = test.drop(TARGET_COL, axis=1)
    y_test  = test[TARGET_COL]

    # ── MLflow Setup ──────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():

        # ── Train ─────────────────────────────────────────────
        model = LinearRegression()
        model.fit(X_train, y_train)
        logger.info("Training completed")

        # ── Evaluate ──────────────────────────────────────────
        preds = model.predict(X_test)
        mse   = mean_squared_error(y_test, preds)
        rmse  = np.sqrt(mse)
        r2    = r2_score(y_test, preds)

        logger.info(f"MSE:  {mse:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R2:   {r2:.4f}")

        # ── Log Parameters to MLflow ──────────────────────────
        mlflow.log_param("model_type",    "LinearRegression")
        mlflow.log_param("random_state",  RANDOM_STATE)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples",  len(X_test))
        mlflow.log_param("features",      X_train.shape[1])

        # ── Log Metrics to MLflow ─────────────────────────────
        mlflow.log_metric("mse",  mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2",   r2)

        # ── Log Model to MLflow ───────────────────────────────
        mlflow.sklearn.log_model(model, "model")

        # ── Save model as pickle (for FastAPI) ────────────────
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved → {MODEL_PATH}")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run_id: {run_id}")

    logger.info("=== MODEL TRAINING DONE ===")
    return model

if __name__ == "__main__":
    train_and_log()