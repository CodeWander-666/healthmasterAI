
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# URL for the Pima Indians Diabetes dataset (UCI)
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_URL, names=COLUMN_NAMES)

    print("Dataset shape:", df.shape)
    print("Class distribution:\n", df['Outcome'].value_counts())

    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split into train/test (optional, just for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create a pipeline: scaling + logistic regression
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
    # After scaling, coefficients indicate importance on normalized scale
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
