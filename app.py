import streamlit as st
import numpy as np
import joblib
from typing import Dict

# CONFIG

APP_TITLE = "Insurance Fraud Detection System"
MODEL_PATH = "ml/artifacts/model.pkl"  # change if needed

st.set_page_config(
    page_title=APP_TITLE,
    layout="centered"
)

# LOAD MODEL (ONCE)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ML INFERENCE (REAL FIX)

def predict_fraud(payload: Dict) -> Dict:
    """
    Runs real ML inference locally inside Hugging Face.
    """

    # Convert input dict → model input
    features = np.array([list(payload.values())])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    reasons = []
    if payload["claim_amount"] > 100000:
        reasons.append("High claim amount")
    if payload["policy_tenure_days"] < 90:
        reasons.append("Short policy tenure")
    if payload["num_previous_claims"] > 2:
        reasons.append("Multiple previous claims")

    if not reasons:
        reasons.append("No major risk factors detected")

    return {
        "eligible": True,
        "fraud": bool(prediction),
        "probability": round(float(probability), 2),
        "reasons": reasons
    }

# UI HEADER

st.title("🛡️ Insurance Fraud Detection System")
st.write(
    "Evaluate insurance claims for **eligibility**, "
    "**fraud risk**, and receive **explainable decisions**."
)

# INPUT MODE

input_mode = st.radio(
    "Choose input method:",
    ["Claim Details (Structured)", "Claim Description (Text)"]
)

submitted = False

# COMMON INPUTS

def common_claim_inputs():
    insurance_type = st.selectbox(
        "Insurance Type", ["health", "vehicle", "life", "finance"]
    )
    policy_type = st.selectbox(
        "Policy Type", ["basic", "premium"]
    )
    incident_type = st.selectbox(
        "Incident Type",
        ["accident", "illness", "theft", "death", "financial_loss"]
    )
    payment_method = st.selectbox(
        "Payment Method", ["cash", "online", "cheque"]
    )
    region = st.selectbox(
        "Region", ["north", "south", "east", "west"]
    )

    claim_amount = st.number_input("Claim Amount", min_value=0, value=50000)
    customer_age = st.number_input("Customer Age", min_value=18, value=35)
    policy_tenure_days = st.number_input("Policy Tenure (days)", min_value=1, value=180)
    num_previous_claims = st.number_input("Previous Claims", min_value=0, value=0)
    days_since_last_claim = st.number_input("Days Since Last Claim", min_value=0, value=200)
    claim_processing_days = st.number_input("Claim Processing Days", min_value=1, value=10)

    return {
        "insurance_type": insurance_type,
        "policy_type": policy_type,
        "incident_type": incident_type,
        "payment_method": payment_method,
        "region": region,
        "claim_amount": claim_amount,
        "customer_age": customer_age,
        "policy_tenure_days": policy_tenure_days,
        "num_previous_claims": num_previous_claims,
        "days_since_last_claim": days_since_last_claim,
        "claim_processing_days": claim_processing_days,
    }

# FORMS

if input_mode == "Claim Description (Text)":
    with st.form("text_claim_form"):
        st.subheader("📝 Claim Description")
        st.text_area(
            "Describe the claim",
            placeholder="Example: Accident occurred last month, claimed 150000 for vehicle repair..."
        )
        input_data = common_claim_inputs()
        submitted = st.form_submit_button("🔍 Evaluate Claim")
else:
    with st.form("structured_claim_form"):
        st.subheader("📋 Claim Details")
        input_data = common_claim_inputs()
        submitted = st.form_submit_button("🔍 Evaluate Claim")

# OUTPUT

if submitted:
    with st.spinner("Analyzing claim..."):
        result = predict_fraud(input_data)

    st.subheader("📊 Decision Result")

    if result["fraud"]:
        st.error("🚨 Fraudulent Claim Detected")
        st.write("**Action:** Investigate / Reject")
    else:
        st.success("✅ Claim Appears Legitimate")
        st.write("**Action:** Approve")

    st.write(f"**Fraud Probability:** `{result['probability']}`")

    st.write("**Risk Factors:**")
    for r in result["reasons"]:
        st.write("•", r)

st.markdown("---")
st.caption("Insurance Fraud Detection System | Hugging Face Streamlit Deployment")
