from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .model import HealthModel

app = FastAPI(title="HealthMaster AI API")

# Enable CORS for GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your GitHub Pages URL
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = HealthModel()

class PatientInput(BaseModel):
    glucose: float
    bmi: float
    age: int
    bp: float

@app.get("/health")
def health_check():
    return {"status": "operational", "version": "1.0.2"}

@app.post("/predict")
async def get_prediction(patient: PatientInput):
    return engine.predict(patient.dict())
