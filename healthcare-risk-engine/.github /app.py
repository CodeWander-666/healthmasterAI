import streamlit as st
import joblib
import numpy as np

# Load the saved engine
model = joblib.load('models/diabetes_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title("🏥 HealthGuard AI: Real-Time Risk Detection")
st.write("Industry-grade predictive analytics for diabetes risk.")

# User Inputs
glu = st.number_input("Glucose Level", value=120)
bmi = st.number_input("BMI Index", value=25.0)
age = st.slider("Patient Age", 1, 100, 30)
bp = st.number_input("Blood Pressure", value=80)

if st.button("Analyze Risk"):
    features = np.array([[0, glu, bp, 0, 0, bmi, 0.5, age]]) # Simplified input vector
    scaled_f = scaler.transform(features)
    prediction = model.predict_proba(scaled_f)[0][1]
    
    st.subheader(f"Risk Score: {round(prediction * 100, 2)}%")
    
    # Proven Results / Interpretability
    if prediction > 0.7:
        st.error("High Risk Detected. Consult a specialist.")
    else:
        st.success("Low Risk Detected.")

st.sidebar.warning("🛡️ **Privacy Note:** This tool uses de-identified data processing. No PII (Personally Identifiable Information) is stored on our servers.")
