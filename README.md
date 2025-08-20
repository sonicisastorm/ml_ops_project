This project turns Hw1 into a complete MLOps pipeline with:

A FastAPI backend that exposes a /predict API.

A Streamlit frontend that connects to the backend.

Dockerized microservices for both the backend and frontend.

Docker Compose orchestration for local development.

Deployment to AWS EC2 with continuous integration and delivery using GitHub Actions and a self-hosted runner.

Backend (FastAPI)

Found in the backend/ directory.

Offers REST API endpoints:

POST /predict, which returns predictions from the trained ML model.

The model is saved in the model/ directory and loaded during runtime.

Frontend (Streamlit)

Found in the frontend/ directory.

Provides an easy-to-use web interface for model inference.

Sends input features to the backend /predict endpoint.

Displays predictions and visualizations.

Runs on port 8501 by default.

Containerization & Orchestration

Both services are containerized with Docker and managed with Docker Compose.

To run everything locally, use:
docker compose up --build

Deployment on AWS EC2

First, launch an EC2 instance running Ubuntu.

Install Docker and Docker Compose.

Clone this repository.

Set up a self-hosted GitHub Actions runner on the EC2 instance.

When you push code, GitHub Actions builds, tests, and deploys automatically.

Workflow runs when there is a push to the main branch.

Jobs:

Linting with ruff.

Note:
In the Assets folder you will find the screenshots
