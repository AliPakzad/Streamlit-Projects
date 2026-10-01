import streamlit as st
import pandas as pd
import joblib
import os

# Find the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the pkl files next to app.py
model_path = os.path.join(BASE_DIR, "xgboost_target_encoding.pkl")
global_mean_path = os.path.join(BASE_DIR, "global_mean.pkl")
address_te_path = os.path.join(BASE_DIR, "address_te_mapping.pkl")

# Load the files
model = joblib.load(model_path)
global_mean = joblib.load(global_mean_path)
address_te_mapping = joblib.load(address_te_path)

# Display a header banner from a URL or local file path
banner_image = "header.png"

if os.path.exists(banner_image):
    st.image(
        banner_image,
        use_container_width=True
    )


st.markdown("""
<style>

/* =========================
   Widget Labels
   ========================= */

div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p,
div[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p {
    font-size: 22px !important;
    font-weight: 600 !important;
}


/* =========================
   Selectbox Label
   ========================= */

div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p {
    font-size: 22px !important;
    font-weight: 600 !important;
}


/* =========================
   Selectbox Selected Value
   ========================= */

div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    font-size: 20px !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    font-size: 20px !important;
}


/* =========================
   Selectbox Dropdown
   ========================= */

/* Dropdown container */
div[data-baseweb="popover"] {
    font-size: 20px !important;
}

/* Dropdown menu */
div[data-baseweb="menu"] {
    font-size: 20px !important;
}

/* Individual options */
div[data-baseweb="menu"] [role="option"] {
    font-size: 20px !important;
    line-height: 1.5 !important;
}

/* Text inside each option */
div[data-baseweb="menu"] [role="option"] * {
    font-size: 20px !important;
}


/* =========================
   Number Input
   ========================= */

div[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p {
    font-size: 22px !important;
    font-weight: 600 !important;
}

div[data-testid="stNumberInput"] input {
    font-size: 20px !important;
    height: 50px;
}


/* =========================
   Input Height
   ========================= */

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    min-height: 50px;
}
 div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] [role="option"],
    ul[role="listbox"] li,
    li[role="option"] {
        font-size: 19px !important;
        line-height: 1.6 !important;
        padding: 10px 16px !important;
}


/* =========================
   Predict Button
   ========================= */

div.stButton > button {
    width: 100% !important;
    height: 60px;
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #10C0FF, #2E86C1);
    color: white;
    border: none;
    border-radius: 14px;
    box-shadow: 0 4px 15px rgba(16, 192, 255, 0.3);
    margin-top: 15px;
}
 .stButton > button p,
    .stButton > button div,
    .stButton > button span {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: white !important;
    }

div.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.40);

    background: linear-gradient(
        90deg,
        #1d4ed8,
        #2563eb
    );
}

div.stButton > button:active {
    transform: translateY(0px);
}

/* =========================
   Checkbox
   ========================= */

.stCheckbox input[type="checkbox"] {
    width: 22px !important;
    height: 22px !important;
    transform: scale(1.2);
    margin-right: 8px;
}


</style>
""", unsafe_allow_html=True)



# Page configuration
st.set_page_config(
    page_title="Tehran House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# App title
#st.title("🏡 House Characteristics")
st.write(
    "##### Enter the house characteristics below to estimate its price."
)

st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)


address = st.selectbox(
    "🏙️ Address",
    options=sorted(address_te_mapping.keys()),
    help="Start typing to search..."
)
st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)


# House area
area = st.number_input(
    "📐 Area (m²)",
    min_value=20,
    max_value=2000,
    value=100,
    step=1,
    help="Enter the property area in square meters"
)
st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)


# Number of rooms
room = st.number_input(
    "🚪 Number of Rooms",
    min_value=0,
    max_value=10,
    value=2,
    step=1
)
st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)
with col1:
    parking = st.checkbox(r"$\textsf{\LARGE🚗 Parking}$")
with col2:
    warehouse = st.checkbox(r"$\textsf{\LARGE📦 Warehouse}$")
with col3:
    elevator = st.checkbox(r"$\textsf{\LARGE🛗 Elevator}$")


# Convert Address to its smoothed Target Encoding value
address_te = address_te_mapping.get(
    address,
    global_mean
)


# Create model input
input_data = pd.DataFrame({
    "Area": [area],
    "Room": [room],
    "Parking": [parking],
    "Warehouse": [warehouse],
    "Elevator": [elevator],
    "Address_TE": [address_te]
})



# --------------------------------------------------
# Prediction button
# --------------------------------------------------

# Make prediction
if st.button("🏠  Predict House Price"):

    prediction = model.predict(input_data)[0]
    prediction_billion = prediction / 1_000_000_000

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(16,192,255,0.15), rgba(46,134,193,0.15));
        border: 2px solid #10C0FF;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin-top: 25px;
    ">
        <p style="color: #10C0FF; font-size: 14px; margin: 0; letter-spacing: 2px; font-weight: 600;">
            💰 ESTIMATED HOUSE PRICE
        </p>
        <p style="color: white; font-size: 32px; font-weight: bold; margin: 12px 0;">
            {prediction:,.0f} Toman
        </p>
        <p style="color: #aaa; font-size: 14px; margin: 0;">
            ≈ {prediction_billion:.2f} Billion Toman
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built by Ali Pakzad · Model: XGBoost + Target Encoding · Data: Divar.ir")

