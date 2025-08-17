import os
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="ML Predictor", layout="wide")

# Use API_URL env var if provided, otherwise default to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
HEALTH_URL = API_URL.replace("/predict", "/health")

st.title("🤖 ML Predictor")
st.write("Enter feature values and get predictions from the ML model.")

# --- Health check button ---
if st.button("Check Backend Health"):
    try:
        resp = requests.get(HEALTH_URL, timeout=10)
        if resp.status_code == 200:
            st.success(f"✅ Backend is healthy: {resp.json()}")
        else:
            st.error(f"⚠️ Backend health check failed: {resp.status_code}")
    except Exception as e:
        st.error(f"❌ Could not connect to backend: {e}")

st.divider()

# --- Prediction form ---
f1 = st.number_input("Feature 1", value=1.0)
f2 = st.number_input("Feature 2", value=2.0)
f3 = st.number_input("Feature 3", value=3.0)

if st.button("Predict"):
    payload = {"feature1": f1, "feature2": f2, "feature3": f3}
    try:
        resp = requests.post(API_URL, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            st.success("Prediction successful!")
            st.write(data)
        else:
            st.error(f"API Error {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
