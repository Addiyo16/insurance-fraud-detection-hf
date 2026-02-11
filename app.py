import streamlit as st
import requests
from typing import Dict

# CONFIGURATION

APP_TITLE = "Insurance Fraud Detection System"
API_URL = "https://insurance-fraud-detection-2-f459.onrender.com/predict"
REQUEST_TIMEOUT = 30

st.set_page_config(
    page_title=APP_TITLE,
    layout="centered"
)

# HELPER FUNCTIONS

def call_fraud_api(payload: Dict) -> Dict:
    """
    Sends claim data to the FastAPI backend and returns the prediction result.
    """
    response = requests.post(
        API_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def render_result(result: Dict) -> None:
    """
    Renders the prediction result returned by the backend.
    """
    st.subheader("📊 Decision Result")

    if not result.get("eligible", False):
        st.error("❌ Claim Not Eligible")
        for reason in result.get("reasons", []):
            st.write("•", reason)
        return

    if result.get("fraud", False):
        st.error("🚨 Fraudulent Claim Detected")
        st.write("**Action:** Investigate / Reject")
    else:
        st.success("✅ Claim Appears Legitimate")
        st.write("**Action:** Approve")

    probability = result.get("probability")
    if probability is not None:
        st.write(f"**Fraud Probability:** `{probability}`")

    reasons = result.get("reasons", [])
    if reasons:
        st.write("**Risk Factors:**")
        for r in reasons:
            st.write("•", r)

# UI HEADER

st.title("🛡️ Insurance Fraud Detection System")
st.write(
    "Submit an insurance claim to evaluate **eligibility**, "
    "**fraud risk**, and receive **explainable decisions**."
)

# INPUT MODE SELECTION

input_mode = st.radio(
    "Choose input method:",
    ["Claim Details (Structured)", "Claim Description (Text)"]
)

submitted = False

# INPUT FORMS

def common_claim_inputs():
    """
    Renders shared claim input fields.
    """
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

    claim_amount = st.number_input(
        "Claim Amount", min_value=0, value=50000
    )
    customer_age = st.number_input(
        "Customer Age", min_value=18, value=35
    )
    policy_tenure_days = st.number_input(
        "Policy Tenure (days)", min_value=1, value=180
    )
    num_previous_claims = st.number_input(
        "Previous Claims", min_value=0, value=0
    )
    days_since_last_claim = st.number_input(
        "Days Since Last Claim", min_value=0, value=200
    )
    claim_processing_days = st.number_input(
        "Claim Processing Days", min_value=1, value=10
    )

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

# API CALL & OUTPUT

if submitted:
    try:
        with st.spinner("Analyzing claim..."):
            result = call_fraud_api(input_data)

        render_result(result)

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Backend may be waking up. Please retry.")

    except requests.exceptions.RequestException as e:
        st.error("⚠️ Unable to reach backend service.")
        st.caption(str(e))
