import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Data
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv')
df.columns = ['Preg', 'Glu', 'BP', 'Skin', 'Ins', 'BMI', 'DPF', 'Age', 'Outcome']

# 1. Normalize (Industry Grade Consistency)
scaler = StandardScaler()
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_scaled = scaler.fit_transform(X)

# 2. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 3. Save for Production
joblib.dump(model, 'models/diabetes_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Model trained and exported successfully!")
