import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Create directory for artifacts
os.makedirs('models', exist_ok=True)

def train_health_model():
    # Dataset: Pima Indians Diabetes (Industry Benchmark)
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
    df = pd.read_csv(url, names=columns)

    # 1. Medical Data Normalization (Handling Zeros as NaNs)
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_fix:
        df[col] = df[col].replace(0, df[col].median())

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 2. Industry-Grade Scaling (Robust to medical outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Model Training
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 4. Proven Results Validation
    print("Clinical Accuracy Report:")
    print(classification_report(y_test, model.predict(X_test)))

    # 5. Export Artifacts
    joblib.dump(model, 'models/health_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(X.columns.tolist(), 'models/features.pkl')
    print("✅ Model & Scaler Exported to /models/")

if __name__ == "__main__":
    train_health_model()
