import streamlit as st
import pandas as pd
import joblib
from textwrap import dedent

# Import the custom transformer used inside the saved pipeline
from feature_engineering import ChurnFeatureEngineer


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- Main App ---------- */

    .main {
        padding-top: 1rem;
    }

    /* ---------- Header ---------- */

    .hero {
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        background: linear-gradient(
            135deg,
            #151922 0%,
            #1d2330 100%
        );
        border: 1px solid #303643;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #aeb6c5;
        margin-bottom: 1rem;
    }

    .model-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        background-color: #252c3a;
        border: 1px solid #3a4354;
        color: #cdd5e3;
        font-size: 0.85rem;
    }


    /* ---------- Section Headers ---------- */

    .section-description {
        color: #9da6b5;
        font-size: 0.9rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }


    /* ---------- Input Labels ---------- */

    label {
        font-weight: 500 !important;
    }


    /* ---------- Expander ---------- */

    .streamlit-expanderHeader {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }


    /* ---------- Predict Button ---------- */

    div.stButton > button {
        width: 100%;
        height: 3.2rem;
        border-radius: 12px;
        border: none;
        font-size: 1.05rem;
        font-weight: 600;
        background: linear-gradient(
            90deg,
            #ff4b4b,
            #ff6b6b
        );
        color: white;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(255, 75, 75, 0.25);
    }


    /* ---------- Prediction Card ---------- */

    .prediction-card {
        margin-top: 1.5rem;
        padding: 2rem;
        border-radius: 18px;
        background: #151922;
        border: 1px solid #303643;
        text-align: center;
    }

    .prediction-label {
        font-size: 0.9rem;
        color: #9da6b5;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .prediction-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .probability {
        font-size: 3.2rem;
        font-weight: 750;
        margin: 0.5rem 0;
    }

    .probability-label {
        color: #aeb6c5;
        font-size: 0.95rem;
    }

    .risk-bar-container {
        width: 100%;
        height: 10px;
        background: #292f3b;
        border-radius: 10px;
        margin: 1.3rem 0 0.8rem 0;
        overflow: hidden;
    }

    .risk-bar {
        height: 100%;
        border-radius: 10px;
    }

    .prediction-message {
        color: #b7bfcc;
        font-size: 0.95rem;
        margin-top: 1rem;
    }


    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #6f7785;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #292f39;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_pipeline():
    return joblib.load("churn_rf_smote_pipeline.pkl")


pipeline = load_pipeline()


# ============================================================
# Header
# ============================================================

st.markdown(dedent("""
<div class="hero">
    <div class="hero-title">
        📉 Telco Customer Churn Predictor
    </div>
    <div class="hero-subtitle">
        Predict the likelihood that a customer will leave the
        telecommunications service based on their profile,
        services, and billing information.
    </div>
    <span class="model-badge">
        🌲 Random Forest &nbsp; • &nbsp; SMOTE
    </span>
</div>
"""), unsafe_allow_html=True)


# ============================================================
# Customer Information
# ============================================================

with st.expander("👤 Customer Information", expanded=True):

    st.markdown(
        '<div class="section-description">'
        'Basic demographic and customer relationship information.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

    with col2:

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )


# ============================================================
# Phone & Internet Services
# ============================================================

with st.expander("📞 Phone & Internet Services"):

    st.markdown(
        '<div class="section-description">'
        'Information about the customer\'s phone and internet services.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

    with col2:

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )


# ============================================================
# Online Services
# ============================================================

with st.expander("🛡️ Online Services"):

    st.markdown(
        '<div class="section-description">'
        'Optional online services and entertainment features.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with col2:

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )


# ============================================================
# Billing & Contract
# ============================================================

with st.expander("💳 Billing & Contract"):

    st.markdown(
        '<div class="section-description">'
        'Contract, billing preferences, and monthly charges.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col2:

        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.5
        )

        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0,
            step=10.0
        )


# ============================================================
# Prepare Input Data
# ============================================================

input_data = pd.DataFrame({

    "gender": [1 if gender == "Male" else 0],
    "SeniorCitizen": [1 if senior == "Yes" else 0],
    "Partner": [1 if partner == "Yes" else 0],
    "Dependents": [1 if dependents == "Yes" else 0],
    "tenure": [tenure],
    "PhoneService": [1 if phone_service == "Yes" else 0],
    "MultipleLines": [1 if multiple_lines == "Yes" else 0],
    "InternetService": [internet_service],
    "OnlineSecurity": [1 if online_security == "Yes" else 0],
    "OnlineBackup": [1 if online_backup == "Yes" else 0],
    "DeviceProtection": [1 if device_protection == "Yes" else 0],
    "TechSupport": [1 if tech_support == "Yes" else 0],
    "StreamingTV": [1 if streaming_tv == "Yes" else 0],
    "StreamingMovies": [1 if streaming_movies == "Yes" else 0],
    "Contract": [contract],
    "PaperlessBilling": [1 if paperless == "Yes" else 0],
    "PaymentMethod": [payment],
    "MonthlyCharges": [monthly],
    "TotalCharges": [total]
})


# ============================================================
# Prediction Section
# ============================================================

st.markdown("### 🔮 Churn Risk Assessment")

st.markdown(
    '<div class="section-description">'
    'Review the information above and run the model to estimate the customer\'s churn risk.'
    '</div>',
    unsafe_allow_html=True
)

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:

    predict_button = st.button(
        "🔮 Predict Churn",
        use_container_width=True
    )


# ============================================================
# Prediction
# ============================================================

if predict_button:

    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]
    probability_percent = probability * 100

    # --------------------------------------------------------
    # High Risk
    # --------------------------------------------------------

    if prediction == 1:

        st.markdown(dedent(f"""
        <div class="prediction-card">
            <div class="prediction-label">
                Churn Prediction
            </div>
            <div class="prediction-title">
                ⚠️ High Risk of Churn
            </div>
            <div class="probability">
                {probability_percent:.1f}%
            </div>
            <div class="probability-label">
                Probability of Churn
            </div>
            <div class="risk-bar-container">
                <div class="risk-bar" style="width: {probability_percent}%; background: #ff4b4b;"></div>
            </div>
            <div class="prediction-message">
                This customer shows a high likelihood of leaving.
                Consider taking retention actions or offering an incentive.
            </div>
        </div>
        """), unsafe_allow_html=True)

    # --------------------------------------------------------
    # Low Risk
    # --------------------------------------------------------

    else:

        st.markdown(dedent(f"""
        <div class="prediction-card">
            <div class="prediction-label">
                Churn Prediction
            </div>
            <div class="prediction-title">
                ✅ Customer Likely to Stay
            </div>
            <div class="probability">
                {probability_percent:.1f}%
            </div>
            <div class="probability-label">
                Probability of Churn
            </div>
            <div class="risk-bar-container">
                <div class="risk-bar" style="width: {probability_percent}%; background: #2ecc71;"></div>
            </div>
            <div class="prediction-message">
                This customer currently shows a relatively low risk
                of churning.
            </div>
        </div>
        """), unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================

st.markdown(dedent("""
<div class="footer">
    Telco Customer Churn Prediction &nbsp;•&nbsp;
    Random Forest with SMOTE
</div>
"""), unsafe_allow_html=True)