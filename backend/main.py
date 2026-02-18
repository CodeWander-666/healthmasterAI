import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'diabetes_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    pipeline = pickle.load(f)

# Feature names in the exact order expected by the model (as in CSV)
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Map frontend keys (lowercase, with "dpf" shorthand) to feature names
KEY_MAPPING = {
    'pregnancies': 'Pregnancies',
    'glucose': 'Glucose',
    'bloodpressure': 'BloodPressure',
    'skinthickness': 'SkinThickness',
    'insulin': 'Insulin',
    'bmi': 'BMI',
    'dpf': 'DiabetesPedigreeFunction',
    'age': 'Age'
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Build feature vector in the correct order using the mapping
        features = []
        for feature in FEATURE_NAMES:
            # Find the frontend key that corresponds to this feature
            frontend_key = None
            for k, v in KEY_MAPPING.items():
                if v == feature:
                    frontend_key = k
                    break
            if frontend_key is None:
                return jsonify({'error': f'No mapping for feature {feature}'}), 500

            if frontend_key not in data:
                return jsonify({'error': f'Missing field: {frontend_key}'}), 400
            try:
                value = float(data[frontend_key])
            except ValueError:
                return jsonify({'error': f'Invalid numeric value for {frontend_key}'}), 400
            features.append(value)

        X = np.array(features).reshape(1, -1)

        prediction = int(pipeline.predict(X)[0])
        probability = pipeline.predict_proba(X)[0][1]

        # Feature importance: coefficients from logistic regression
        lr_model = pipeline.named_steps['logisticregression']
        coef = lr_model.coef_[0]

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
