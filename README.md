# ML Ops Project

![CI/CD](https://github.com/sonicisastorm/ml_ops_project/actions/workflows/ci-build.yaml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

A full end-to-end MLOps pipeline featuring a **FastAPI** prediction backend, a **Streamlit** frontend, **Docker** containerization, and automated **CI/CD** deployment to **AWS EC2** via GitHub Actions.

---

## Architecture Overview

```
┌─────────────────────┐        HTTP        ┌──────────────────────┐
│  Streamlit Frontend │  ──── /predict ──►  │  FastAPI Backend     │
│     (port 8501)     │                     │     (port 8000)      │
└─────────────────────┘                     └──────────┬───────────┘
                                                       │ joblib.load
                                                ┌──────▼───────┐
                                                │  model.pkl   │
                                                └──────────────┘
```

Both services run inside Docker containers orchestrated by Docker Compose, and are deployed to an AWS EC2 instance through a GitHub Actions CI/CD pipeline.

---

## Project Structure

```
ml_ops_project-main/
├── .github/
│   └── workflows/
│       └── ci/cd-cd.yaml       # GitHub Actions CI/CD pipeline
├── assets/                     # Screenshots and project images
├── backend/
│   ├── src/
│   │   ├── data/
│   │   │   └── make_dataset.py         # Data loading & preprocessing
│   │   ├── features/
│   │   │   └── build_features.py       # Feature engineering
│   │   ├── models/
│   │   │   ├── train_model.py          # Model training script
│   │   │   └── predict_model.py        # Inference script
│   │   └── visualization/
│   │       └── visualize.py            # Plotting utilities
│   ├── notebooks/
│   │   ├── ali-gasimov-eda.ipynb       # Exploratory data analysis
│   │   ├── ali-gasimov-fe1.ipynb       # Feature engineering (part 1)
│   │   ├── ali-gasimov-fe2.ipynb       # Feature engineering (part 2)
│   │   └── ali-gasimov-trainmodel.ipynb# Model training notebook
│   ├── data/
│   │   ├── external/                   # Raw data from third-party sources
│   │   ├── interim/                    # Intermediate transformed data
│   │   └── processed/                  # Final datasets ready for modeling
│   ├── app.py                          # FastAPI application
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py                          # Streamlit application
│   ├── Dockerfile
│   └── requirements.txt
├── models/
│   └── model.pkl                       # Trained model artifact
├── docker-compose.yml
├── pyproject.toml
└── LICENSE
```

---

## Services

### Backend (FastAPI)

The backend loads a pre-trained model at startup and exposes two endpoints:

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | `/health`  | Returns service status and UTC time |
| POST   | `/predict` | Accepts feature JSON, returns predictions |

**Example request:**
```json
POST /predict
{
    "feature1": 3.5,
    "feature2": 1.2,
    "feature3": 0.7
}
```

**Example response:**
```json
{
    "status": "success",
    "predictions": [42.7],
    "num_predictions": 1
}
```

Interactive API docs are available at `http://localhost:8000/docs` when running locally.

### Frontend (Streamlit)

A simple web UI running on port `8501` that lets you:
- Check the backend health status
- Input feature values via number fields
- Send requests to `/predict` and display the result

The frontend connects to the backend via the `API_URL` environment variable (defaults to `http://localhost:8000/predict`).

---

## Datasets

The project uses two datasets stored in AWS S3:

- **Amsterdam Housing Prices** (`HousingPrices-Amsterdam-August-2021.csv`) — used for the primary regression task
- **Ramen Ratings** (`ramen-ratings.csv`) — used for supplementary analysis

---

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/)
- Python 3.10+ (for local development without Docker)

### Run with Docker Compose (recommended)

```bash
git clone https://github.com/sonicisastorm/ml_ops_project.git
cd ml_ops_project-main

docker compose up --build
```

- Frontend: http://localhost:8501  
- Backend API: http://localhost:8000  
- API Docs: http://localhost:8000/docs

### Run Locally (without Docker)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### Train the Model

```bash
cd backend
uv sync
uv run python -m src.models.train_model
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci/cd-cd.yaml`) triggers on every push to `main` and runs on a **self-hosted runner** provisioned on EC2:

1. **Checkout** — pulls the latest code
2. **Lint** — runs `ruff check .` to enforce code quality
3. **Build** — builds both `ml-backend` and `ml-frontend` Docker images
4. **Deploy** — runs `docker compose up -d --build` on the EC2 instance

### Setting Up the EC2 Runner

1. Launch an Ubuntu EC2 instance
2. Install Docker and Docker Compose
3. Clone this repository onto the instance
4. Register a [self-hosted GitHub Actions runner](https://docs.github.com/en/actions/hosting-your-own-runners/adding-self-hosted-runners) on the instance
5. Push to `main` — deployments will happen automatically from that point on

---

## Code Quality

The project uses `ruff`, `black`, and `isort` for linting and formatting. Run them via `uvx` (no extra dev dependencies needed):

```bash
# Lint
uvx ruff check .

# Format
uvx isort .
uvx black .

# Auto-fix
uvx ruff check --fix .
```

---

## Screenshots

Screenshots of the running application are available in the `assets/` directory.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
