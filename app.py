import os
import joblib
import streamlit as st
import pandas as pd

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Insurance Fraud Detection System")

# --------------------------------------------------
# PATH (ROOT FOLDER)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_fraud_pipeline.pkl")

# --------------------------------------------------
# LOAD MODEL (PIPELINE ONLY)
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()
st.success("✅ Model loaded successfully")

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
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

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🔍 Predict Fraud"):
    input_df = pd.DataFrame([{
        "insurance_type": insurance_type,
        "policy_type": policy_type,
        "incident_type": incident_type,
        "payment_method": payment_method,
        "region": region,
        "claim_amount": claim_amount
    }])

    prediction = model.predict(input_df)[0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraudulent Claim Detected")
    else:
        st.success("✅ Genuine Claim")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit & ML Pipeline")


