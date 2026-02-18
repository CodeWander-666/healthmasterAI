"""
Model training script for diabetes prediction using provided CSV.
Trains a logistic regression model with scaling and saves the pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Path to the CSV file (assumed to be in the same directory as this script)
CSV_PATH = os.path.join(os.path.dirname(__file__), 'Diabetes_prediction.csv')

def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    # Check for any missing or invalid values (optional cleaning)
    # For simplicity, we use the data as is; note that some values may be unrealistic.
    # In production, you should handle zeros/negatives appropriately.

    print("Dataset shape:", df.shape)
    print("Class distribution:\n", df['Diagnosis'].value_counts())

    # Separate features and target
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create pipeline: scaling + logistic regression
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=1000)
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

    # Feature importance (coefficients)
    feature_names = X.columns.tolist()
    coef = pipeline.named_steps['logisticregression'].coef_[0]
    print("\nFeature Coefficients (importance):")
    for name, c in zip(feature_names, coef):
        print(f"{name}: {c:.4f}")

    # Save the pipeline
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'diabetes_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()
