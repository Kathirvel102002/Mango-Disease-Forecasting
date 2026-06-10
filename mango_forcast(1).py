import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Mango Disease Forecast System",
    page_icon="🥭",
    layout="wide"
)

st.title("🥭 Mango Disease Forecast System")
st.markdown("### Forecast Next Week Disease Severity")      

# ==========================================================
# INPUTS
# ==========================================================

disease = st.selectbox(
    "Select Disease",
    [
        "Leaf Anthracnose",
        "Black Banded",
        "Red Rust",
        "Die Back",
        "Sooty Mould"
    ]
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Weather Parameters")

    rf = st.number_input(
        "Rainfall (RF) mm",
        min_value=0.0,
        value=25.0
    )

    rd = st.number_input(
        "Rainy Days (RD)",
        min_value=0,
        value=2
    )

    rh = st.number_input(
        "Humidity (RH) %",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

    tmax = st.number_input(
        "Maximum Temperature (°C)",
        value=34.0
    )

    tmin = st.number_input(
        "Minimum Temperature (°C)",
        value=25.0
    )

with col2:
    st.subheader("Field Parameters")

    current_disease = st.number_input(
        "Current Disease Severity (%)",
        min_value=0.0,
        max_value=100.0,
        value=15.0
    )

    week = st.number_input(
        "Week Number (1-52)",
        min_value=1,
        max_value=52,
        value=24
    )

# ==========================================================
# MODEL FILES
# ==========================================================

model_files = {
    "Leaf Anthracnose": "LEAF ANTHRACNOSE_farmer_forecast.pkl",
    "Black Banded": "BLACK_BANDED_farmer_forecast.pkl",
    "Red Rust": "RED RUST_farmer_forecast.pkl",
    "Die Back": "DIE BACK_farmer_forecast.pkl",
    "Sooty Mould": "SOOTY MOULD_farmer_forecast.pkl"
}

# ==========================================================
# PREDICTION
# ==========================================================

if st.button("Forecast Disease Risk"):

    try:

        T_avg = (tmax + tmin) / 2
        T_Range = tmax - tmin

        week_sin = np.sin(2 * np.pi * week / 52)
        week_cos = np.cos(2 * np.pi * week / 52)

        input_data = pd.DataFrame([[
            rf,
            rd,
            rh,
            tmax,
            tmin,
            T_avg,
            T_Range,
            current_disease,
            week_sin,
            week_cos
        ]], columns=[
            "RF",
            "RD",
            "RH",
            "T_MAX",
            "T_MIN",
            "T_avg",
            "T_Range",
            "DISEASE",
            "week_sin",
            "week_cos"
        ])

        model_file = model_files[disease]

        if not os.path.exists(model_file):
            st.error(f"Model file not found: {model_file}")
            st.stop()

        model = joblib.load(model_file)

        forecast = float(model.predict(input_data)[0])

        if forecast < 20:
            risk = "🟢 Low"
            advice = "Routine monitoring is sufficient."

        elif forecast < 40:
            risk = "🟡 Moderate"
            advice = "Inspect orchard regularly."

        elif forecast < 60:
            risk = "🟠 High"
            advice = "Start disease management measures."

        else:
            risk = "🔴 Epidemic"
            advice = "Immediate control measures required."

        st.success("Forecast Completed")

        st.metric(
            label="Predicted Disease Severity (%)",
            value=f"{forecast:.2f}"
        )

        st.subheader("Risk Assessment")
        st.write(f"**Risk Level:** {risk}")
        st.write(f"**Recommendation:** {advice}")

    except Exception as e:
        st.error(str(e))