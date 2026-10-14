import streamlit as st
import pandas as pd
import sklearn
import numpy as np
import joblib
import json

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Developer Salary Predictor",
    page_icon="💰",
    layout="wide"
)

# ============================================================
# LOAD MODEL AND METADATA (cached for performance)
# ============================================================
@st.cache_resource
def load_pipeline():
    return joblib.load('salary_predictor_pipeline.pkl')

@st.cache_data
def load_metadata():
    with open('model_metadata.json', 'r') as f:
        return json.load(f)

pipeline = load_pipeline()
meta = load_metadata()

# ============================================================
# FEATURE ENGINEERING HELPERS
# ============================================================
def org_size_to_numeric(value):
    """Map organization size category to its lower-bound numeric value."""
    mapping = {
        'Just me - I am a freelancer, sole proprietor, etc.': 1,
        '2 to 9 employees': 2,
        '10 to 19 employees': 10,
        '20 to 99 employees': 20,
        '100 to 499 employees': 100,
        '500 to 999 employees': 500,
        '1,000 to 4,999 employees': 1000,
        '5,000 to 9,999 employees': 5000,
        '10,000 or more employees': 10000,
    }
    return mapping.get(value, np.nan)

def age_to_ordinal(age_str):
    """Map age category to an ordinal integer."""
    mapping = {
        '18-24 years old': 1,
        '25-34 years old': 2,
        '35-44 years old': 3,
        '45-54 years old': 4,
        '55-64 years old': 5,
        '65 years or older': 6,
    }
    return mapping.get(age_str, np.nan)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #00695C, #00897B); 
            padding:25px; border-radius:15px; 
            box-shadow:0px 6px 20px rgba(0,0,0,0.15); margin-bottom:20px;">
    <h1 style="color:white; text-align:center; margin:0; font-size:36px;">
        💰 Developer Salary Predictor
    </h1>
    <p style="color:white; text-align:center; margin-top:10px; font-size:16px;">
        Based on the Stack Overflow 2022 Developer Survey · CatBoost model
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT FORM
# ============================================================
st.markdown("### 📝 Enter your professional profile")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👤 Personal")
    age = st.selectbox("Age Group", meta['age_categories'], index=1)
    gender = st.selectbox("Gender", meta['gender_categories'])
    edlevel = st.selectbox("Education Level", meta['edlevel_categories'])
    country = st.selectbox("Country", meta['country_categories'])

with col2:
    st.markdown("#### 💼 Professional")
    years_code_pro = st.slider(
        "Years of Professional Coding",
        min_value=0, max_value=50, value=5, step=1
    )
    org_size = st.selectbox("Organization Size", meta['orgsize_categories'], index=4)
    remote_work = st.selectbox("Remote Work", meta['remotework_categories'])
    main_branch = st.selectbox("Main Branch", meta['mainbranch_categories'])
    icorpm = st.selectbox("Role Type", meta['icorpm_categories'])
    tbranch = st.selectbox("Works in a Tech Branch/Dept?", meta['tbranch_categories'])

# ============================================================
# PREDICTION
# ============================================================
if st.button("🔮 Predict My Salary", use_container_width=True, type="primary"):

    # Build the input DataFrame with the exact feature names used in training
    user_input = pd.DataFrame({
        'YearsCodePro': [years_code_pro],
        'OrgSize_Log': [np.log1p(org_size_to_numeric(org_size))],
        'Age_Ordinal': [age_to_ordinal(age)],
        'Country_Reduced': [country],
        'EdLevel': [edlevel],
        'MainBranch': [main_branch],
        'RemoteWork': [remote_work],
        'Gender': [gender],
        'TBranch': [tbranch],
        'ICorPM': [icorpm],
    })

    # Predict
    prediction = pipeline.predict(user_input)[0]

    # Display result
    st.markdown(f"""
    <div style="background-color:#E8F5E9; padding:25px; border-radius:12px; 
                border-left: 6px solid #00695C; margin-top:20px;">
        <h2 style="color:#00695C; margin:0;">💵 Estimated Annual Salary</h2>
        <p style="font-size:42px; font-weight:bold; color:#1B5E20; margin:15px 0;">
            ${prediction:,.0f}
        </p>
        <p style="color:#555; font-size:14px; margin:0;">
            Predicted for a {icorpm.lower()} in {country}, 
            with {years_code_pro} years of professional coding experience.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This model was trained on salaries between "
        "10,000 and 300,000 USD. Predictions outside this range may not be reliable. "
        f"Model performance: R² = {meta['model_metrics']['r2']:.3f}, "
        f"MAE = ${meta['model_metrics']['mae']:,}."
    )