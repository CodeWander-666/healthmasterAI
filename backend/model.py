import joblib
import numpy as np
import os

class HealthModel:
    def __init__(self):
        # Paths relative to this file
        base_path = os.path.dirname(__file__)
        self.model = joblib.load(os.path.join(base_path, "../models/health_model.pkl"))
        self.scaler = joblib.load(os.path.join(base_path, "../models/scaler.pkl"))

    def predict(self, data):
        # Industry Standard: Map 4 user inputs to the 8-feature clinical vector
        # [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Pedigree, Age]
        clinical_vector = np.array([[1, data['glucose'], data['bp'], 20, 80, data['bmi'], 0.5, data['age']]])
        
        # Normalize and Predict
        scaled_data = self.scaler.transform(clinical_vector)
        risk_probability = self.model.predict_proba(scaled_data)[0][1]
        
        return {
            "risk_score": round(float(risk_probability * 100), 2),
            "triage": "URGENT" if risk_probability > 0.7 else "MONITOR" if risk_probability > 0.4 else "STABLE",
            "audit_id": os.urandom(4).hex() # Traceability for startup audits
        }
