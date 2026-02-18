import streamlit as st
from engine import HealthEngine
import pandas as pd

# Page Config
st.set_page_config(page_title="HealthGuard AI", page_icon="🏥", layout="wide")

# Initialize Engine
@st.cache_resource
def load_engine():
    return HealthEngine()

try:
    engine = load_engine()
except:
    st.error("Model artifacts not found. Please run `python train.py` first.")
    st.stop()

# UI Layout
st.title("🏥 HealthGuard AI: Clinical Risk Prediction")
st.markdown("---")

with st.sidebar:
    st.header("Patient Vitals")
    st.info("Input de-identified patient data for real-time analysis.")
    
    # Inputs
    preg = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose (mg/dL)", 40, 250, 120)
    bp = st.slider("Blood Pressure (mm Hg)", 40, 140, 80)
    skin = st.number_input("Skin Thickness (mm)", 5, 100, 20)
    insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
    bmi = st.number_input("BMI Index", 10.0, 70.0, 25.0)
    pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.slider("Age", 1, 120, 30)

# Prediction Logic
if st.button("Generate Risk Assessment"):
    data = [preg, glucose, bp, skin, insulin, bmi, pedigree, age]
    result = engine.predict_risk(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Probability Risk Score", f"{result['risk_score']}%")
        if result['status'] == "High":
            st.error("Priority Status: URGENT TRIAGE")
        elif result['status'] == "Moderate":
            st.warning("Priority Status: MONITORING REQUIRED")
        else:
            st.success("Priority Status: STABLE")

    with col2:
        st.subheader("Key Risk Drivers (Explainable AI)")
        factors = pd.DataFrame(result['contributing_factors'].items(), columns=['Feature', 'Impact'])
        st.bar_chart(factors.set_index('Feature'))

st.markdown("---")
st.caption("🛡️ **Privacy Compliance:** No Patient Health Information (PHI) is persisted. This tool is for clinical decision support and does not replace medical advice.")
