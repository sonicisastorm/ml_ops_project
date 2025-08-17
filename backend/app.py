import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FastAPI Backend server for ML project",
    description="REST API for ML project",
    version="1.0.0",
    docs_url="/docs",
)

# CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "healthy", "time": datetime.now(timezone.utc).isoformat()}


@app.post("/predict")
def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example request body (JSON):
    {
        "feature1": 3.5,
        "feature2": 1.2,
        "feature3": 0.7
    }
    """
    try:
        # Ensure consistent feature order
        X = pd.DataFrame([features])
        preds = model.predict(X)

        return {
            "status": "success",
            "predictions": preds.tolist(),
            "num_predictions": len(preds),
        }

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
