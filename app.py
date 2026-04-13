from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load model and preprocessors
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Feature names for reference
FEATURE_NAMES = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict failure type based on sensor readings.

    Expected JSON body:
    {
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0
    }
    """
    try:
        data = request.get_json()

        # Validate input
        required_fields = ['air_temperature', 'process_temperature',
                          'rotational_speed', 'torque', 'tool_wear']

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400

        # Create feature array in correct order
        features = np.array([[
            data['air_temperature'],
            data['process_temperature'],
            data['rotational_speed'],
            data['torque'],
            data['tool_wear']
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)

        # Decode prediction
        failure_type = label_encoder.inverse_transform(prediction)[0]

        # Get confidence scores for all classes
        class_names = label_encoder.classes_
        confidence_scores = {
            class_name: float(prob)
            for class_name, prob in zip(class_names, prediction_proba[0])
        }

        return jsonify({
            'success': True,
            'prediction': failure_type,
            'confidence': float(max(prediction_proba[0])),
            'all_probabilities': confidence_scores
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'classes': label_encoder.classes_.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
