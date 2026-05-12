# 🏔️ Landslide Detection MLOps Platform

An end-to-end MLOps pipeline for housing price prediction demonstrating
the complete ML lifecycle — from data ingestion to model monitoring.

---

# 🚀 Features

- 📥 **Automated Data Ingestion** — loads and splits data automatically
- ⚙️ **Data Preprocessing** — handles missing values and encoding
- 🧪 **Experiment Tracking** — MLflow logs metrics, params, and models
- 📊 **Model Monitoring** — Evidently detects data drift and quality issues
- 🚀 **REST API** — FastAPI prediction endpoint with Swagger docs
- 🔄 **CI/CD Pipeline** — GitHub Actions runs pipeline on every push
- 🐳 **Dockerized** — fully containerized for deployment

---

# 🏗️ System Architecture

```text
Raw Data (CSV)
    ↓
Data Ingestion → train.csv / test.csv
    ↓
Preprocessing → train_processed.csv / test_processed.csv
    ↓
Model Training → LinearRegression + MLflow Tracking
    ↓
Model Saved → models/model.pkl
    ↓
Monitoring → Evidently drift + quality reports
    ↓
FastAPI → /predict endpoint
```

---

# 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Pipeline Orchestration | Python |
| Experiment Tracking | MLflow |
| Model Monitoring | Evidently AI |
| Backend API | FastAPI |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| ML Library | Scikit-learn |
| Language | Python 3.11 |

---

# 📂 Project Structure

```text
landslide-mlops-platform/
│
├── src/
│   ├── config.py              # centralised settings
│   ├── logger.py              # logging setup
│   ├── data_ingestion.py      # load and split data
│   ├── data_preprocessing.py  # clean and encode
│   ├── model_trainer.py       # train + MLflow tracking
│   └── monitor.py             # Evidently reports
│
├── pipeline/
│   └── run_pipeline.py        # runs all steps in order
│
├── app/
│   └── main.py                # FastAPI prediction API
│
├── data/
│   ├── raw/                   # raw CSVs
│   └── processed/             # preprocessed CSVs
│
├── models/                    # saved model.pkl
├── reports/                   # Evidently HTML reports
├── logs/                      # pipeline logs
├── mlruns/                    # MLflow experiment data
│
├── .github/workflows/ci.yml   # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

# ⚙️ Setup & Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/likithamandlem/landslide-mlops-platform.git
cd landslide-mlops-platform
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Full Pipeline

```bash
python pipeline/run_pipeline.py
```

This runs all 4 steps automatically:
1. Data Ingestion
2. Preprocessing
3. Model Training + MLflow logging
4. Evidently monitoring reports

---

# 📊 View MLflow UI

```bash
python -m mlflow ui --backend-store-uri mlruns
```

Open: http://127.0.0.1:5000

---

# 🚀 Run FastAPI

```bash
uvicorn app.main:app --reload --port 8002
```

Open: http://127.0.0.1:8002/docs

## Example Prediction Request

```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41.0,
  "total_rooms": 880.0,
  "total_bedrooms": 129.0,
  "population": 322.0,
  "households": 126.0,
  "median_income": 8.3252,
  "ocean_proximity_NEAR_BAY": 1.0
}
```

## Example Response

```json
{
  "predicted_house_value": 415721,
  "model_version": "LinearRegression-v1",
  "currency": "USD"
}
```

---

# 🐳 Docker Setup

```bash
docker-compose up --build
```

- API: http://localhost:8002/docs
- MLflow: http://localhost:5000

---

# 📈 MLflow Metrics (Latest Run)

| Metric | Value |
|---|---|
| MSE | 4,904,364,240 |
| RMSE | 70,031 |
| R2 Score | 0.626 |

---

# 🧩 Challenges Faced

- Fixing hardcoded `/content/` Google Colab paths for local execution
- MLflow Windows path conflict — solved using `.as_uri()` for file URIs
- Evidently incompatibility with NumPy 2.0 — pinned to NumPy < 2.0
- Aligning train and test feature columns after one-hot encoding

---

# 🔮 Future Improvements

- Add RandomForest and XGBoost — compare runs in MLflow
- Migrate to cloud storage (AWS S3) for data and model artifacts
- Add automated retraining trigger on data drift detection
- Deploy to AWS ECS with auto-scaling
- Add Grafana + Prometheus for real-time monitoring

---

# 🎯 Target Roles

- MLOps Engineer
- ML Platform Engineer
- AI Engineer
- Data Engineer

---

# 👩‍💻 Author

**Likitha Mandlem**
GitHub: https://github.com/likithamandlem