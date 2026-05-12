# src/logger.py
import logging
import os
from datetime import datetime
from src.config import LOGS_DIR

LOG_FILE = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
LOG_PATH = LOGS_DIR / LOG_FILE

logging.basicConfig(
    filename=str(LOG_PATH),
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    level=logging.INFO
)

# Also print logs to terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("[ %(asctime)s ] %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)

def get_logger(name: str = "mlops"):
    return logging.getLogger(name)