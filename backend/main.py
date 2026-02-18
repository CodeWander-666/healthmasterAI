import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Load the pre-trained pipeline (model + scaler)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'diabetes_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    pipeline = pickle.load(f)

# Feature names in the exact order expected by the model
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON with keys matching FEATURE_NAMES (lowercase).
    Returns prediction, probability, and feature importance.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Build feature vector in the correct order
        features = []
        for name in FEATURE_NAMES:
            key = name.lower()  # JSON keys are lowercase in frontend
            if key not in data:
                return jsonify({'error': f'Missing field: {key}'}), 400
            try:
                value = float(data[key])
            except ValueError:
                return jsonify({'error': f'Invalid numeric value for {key}'}), 400
            features.append(value)

        # Convert to numpy array and reshape for single sample
        X = np.array(features).reshape(1, -1)

        # Predict
        prediction = int(pipeline.predict(X)[0])
        probability = pipeline.predict_proba(X)[0][1]  # probability of class 1

        # Feature importance: coefficients from logistic regression
        # Access the logistic regression step inside the pipeline
        # Pipeline steps: [('standardscaler', ...), ('logisticregression', ...)]
        lr_model = pipeline.named_steps['logisticregression']
        coef = lr_model.coef_[0]  # shape (n_features,)

        # Prepare response
        response = {
            'prediction': prediction,
            'probability': probability,
            'importance': coef.tolist(),
            'feature_names': FEATURE_NAMES
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
