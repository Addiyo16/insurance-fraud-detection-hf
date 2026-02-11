import os
import joblib
import streamlit as st
import pandas as pd

# PAGE CONFIG

st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Insurance Fraud Detection System")
st.write("Predict whether an insurance claim is **fraudulent or genuine**.")

# PATH SETUP (DEPLOYMENT SAFE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml", "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_fraud_pipeline")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "encoder.pkl")

# LOAD MODEL & ARTIFACTS (NO SPINNER HERE)

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()

    model = joblib.load(MODEL_PATH)

    scaler = None
    encoder = None

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

    if os.path.exists(ENCODER_PATH):
        encoder = joblib.load(ENCODER_PATH)

    return model, scaler, encoder


model, scaler, encoder = load_artifacts()

st.success("✅ Model loaded successfully")

# USER INPUT FORM

st.subheader("📋 Enter Claim Details")

insurance_type = st.selectbox(
    "Insurance Type",
    ["Life", "Health", "Vehicle", "Travel", "Property"]
)

policy_type = st.selectbox(
    "Policy Type",
    ["Basic", "Standard", "Premium"]
)

incident_type = st.selectbox(
    "Incident Type",
    ["Accident", "Theft", "Medical Claim", "Death Claim", "Fire"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["Bank Transfer", "Cheque", "Cash", "UPI"]
)

region = st.selectbox(
    "Region",
    ["Urban", "Rural", "Semi-Urban"]
)

claim_amount = st.number_input(
    "Claim Amount",
    min_value=1000,
    step=500
)

# PREDICTION

if st.button("🔍 Predict Fraud"):
    input_data = pd.DataFrame([{
        "insurance_type": insurance_type,
        "policy_type": policy_type,
        "incident_type": incident_type,
        "payment_method": payment_method,
        "region": region,
        "claim_amount": claim_amount
    }])

    # Optional preprocessing
    if encoder:
        input_data = encoder.transform(input_data)

    if scaler:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraudulent Claim Detected")
    else:
        st.success("✅ Genuine Claim")

