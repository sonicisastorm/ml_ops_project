This project extends Hw1 into a full MLOps pipeline, with:

A FastAPI backend exposing a /predict API.

A Streamlit frontend that interacts with the backend.

Dockerized microservices for backend & frontend.

Docker Compose orchestration for local development.

Deployment to AWS EC2 with CI/CD using GitHub Actions and a self-hosted runner.

Backend (FastAPI)

Located in backend/

Exposes REST API endpoints:

POST /predict → returns prediction from the trained ML model.

Model is serialized (saved in model/) and loaded at runtime.

Frontend (Streamlit)

Located in frontend/

Provides a user-friendly web interface for model inference.

Sends input features to backend /predict endpoint.

Displays predictions + visualizations.

Runs on port 8501 by default.


Containerization & Orchestration

Both services are containerized with Docker and orchestrated with Docker Compose.


Run everything locally:
docker compose up --build

Deployment on AWS EC2

Launch an EC2 instance (Ubuntu).

Install Docker & Docker Compose.

Clone this repository.

Set up a self-hosted GitHub Actions runner on the EC2 instance.

Push code → GitHub Actions builds, tests, and deploys automatically.


Workflow runs on push to main branch.

Jobs:

Linting (with ruff)

Docker image build (backend + frontend)

Deploy to EC2 (via self-hosted runner)
