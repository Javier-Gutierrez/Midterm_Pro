
# Home Credit Default Risk - Credit Scoring API

End-to-end credit risk scoring service powered by **Logistic Regression**, built with **FastAPI**, managed via **uv**, and containerized with **Docker**.

## Project Architecture

```text
├── data/               # Raw Kaggle datasets (excluded from version control)
├── models/             # Serialized model pipelines (.joblib)
├── src/
│   ├── train.py        # Reproducible preprocessing and training pipeline
│   └── main.py         # FastAPI application and Pydantic validation schemas
├── tests/
│   └── test_api.py     # Automated integration tests using pytest
├── Dockerfile          # Optimized production container configuration
├── pyproject.toml      # Project metadata and dependencies
└── uv.lock             # Deterministic dependency lockfile
```

## Model Overview & Performance

* **Dataset:** Home Credit Default Risk (`application_train.csv`).
* **Preprocessing:** `ColumnTransformer` pipeline handling missing values via `SimpleImputer` (median for numerical, most frequent for categorical), feature scaling via `StandardScaler`, and categorical encoding via `OneHotEncoder`.
* **Algorithm:** Logistic Regression with `class_weight="balanced"` to address class imbalance.
* **Validation Metric (ROC-AUC):** `0.7285`

## Getting Started

### Prerequisites

* [Docker Desktop](https://docker.com) (for containerized execution)
* [uv](https://astral.sh) (for local development and training)
* Download `application_train.csv` from Kaggle and place it inside the `data/` directory.

### Option 1: Run with Docker (Recommended)

1. **Train the model locally** to generate the serialized pipeline artifact:

   ```bash
   uv sync
   uv run python src/train.py
   ```
2. **Build and run the Docker container:**

   ```bash
   docker build -t home-credit-api:v1 .
   docker run -p 8000:8000 home-credit-api:v1
   ```
3. **Access the interactive API documentation (Swagger UI):**
   Open `http://127.0.0` in your browser.

### Option 2: Local Development with `uv`

1. **Install dependencies (including dev tools):**

   ```bash
   uv sync
   ```
2. **Train the model:**

   ```bash
   uv run python src/train.py
   ```
3. **Run automated tests:**

   ```bash
   uv run python -m pytest
   ```
4. **Start the development server:**

   ```bash
   uv run uvicorn src.main:app --reload
   ```

## API Endpoints

* `GET /health`: Health check endpoint confirming server status and model availability.
* `POST /predict`: Evaluates a credit application payload and returns the default probability (`0.0` to `1.0`), binary risk prediction (`0` or `1`), and business risk tier (`Low`, `Medium`, `High`).
