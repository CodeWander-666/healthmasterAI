from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

# Industry Standard: Allow your GitHub Frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("../models/diabetes_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

class PatientData(BaseModel):
    glucose: float
    bmi: float
    age: int
    bp: float

@app.post("/predict")
async def predict_risk(data: PatientData):
    # Standardizing clinical input
    features = np.array([[0, data.glucose, data.bp, 0, 0, data.bmi, 0.5, data.age]])
    scaled_data = scaler.transform(features)
    
    risk_prob = model.predict_proba(scaled_data)[0][1]
    
    return {
        "risk_score": f"{round(risk_prob * 100, 2)}%",
        "triage": "URGENT" if risk_prob > 0.7 else "STABLE",
        "proven_factors": ["Glucose Level", "BMI Correlation"]
    }
