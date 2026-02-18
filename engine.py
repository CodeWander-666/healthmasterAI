import joblib
import numpy as np
import shap
import pandas as pd

class HealthEngine:
    def __init__(self):
        self.model = joblib.load('models/health_model.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        self.features = joblib.load('models/features.pkl')
        self.explainer = shap.TreeExplainer(self.model)

    def predict_risk(self, input_data):
        """
        input_data: list or dict of clinical features
        """
        # Convert to DataFrame for consistency
        df_input = pd.DataFrame([input_data], columns=self.features)
        
        # Normalize
        scaled_input = self.scaler.transform(df_input)
        
        # Probability
        prob = self.model.predict_proba(scaled_input)[0][1]
        
        # SHAP Explainability (The 'Why')
        shap_values = self.explainer.shap_values(scaled_input)
        
        # Handle SHAP output format (differs by version)
        if isinstance(shap_values, list):
            actual_shap = shap_values[1][0] # Class 1 (Positive)
        else:
            actual_shap = shap_values[0][:, 1]

        return {
            "risk_score": round(float(prob * 100), 2),
            "contributing_factors": dict(zip(self.features, actual_shap)),
            "status": "High" if prob > 0.6 else "Moderate" if prob > 0.3 else "Low"
        }
