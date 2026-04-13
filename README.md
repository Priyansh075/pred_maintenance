# Predictive Maintenance Web Application

A machine learning-powered web application that predicts equipment failure types based on sensor readings.

## Project Structure

```
pred_maintenance/
├── app.py                 # Flask backend API
├── model.pkl              # Trained Random Forest model
├── scaler.pkl             # Feature scaler
├── label_encoder.pkl      # Label encoder for classes
├── requirements.txt       # Python dependencies
├── pred_m.ipynb           # Jupyter notebook with model training
└── client/                # React frontend
    ├── package.json
    ├── public/
    │   └── index.html
    └── src/
        ├── index.js
        ├── index.css
        └── App.js
```

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Node.js Dependencies

```bash
cd client
npm install
```

## Running the Application

### Start Flask Backend (Terminal 1)

```bash
python app.py
```

The API will run on `http://localhost:5000`

### Start React Frontend (Terminal 2)

```bash
cd client
npm start
```

The frontend will run on `http://localhost:3000`

## API Endpoints

### POST /api/predict

Predict failure type based on sensor readings.

**Request Body:**
```json
{
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1551,
  "torque": 42.8,
  "tool_wear": 0
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "No Failure",
  "confidence": 0.85,
  "all_probabilities": {
    "No Failure": 0.85,
    "Heat Dissipation Failure": 0.10,
    ...
  }
}
```

### GET /api/health

Health check endpoint.

## Model Details

- **Algorithm:** Random Forest Classifier
- **Classes:** 6 failure types
  - No Failure
  - Heat Dissipation Failure
  - Overstrain Failure
  - Power Failure
  - Random Failures
  - Tool Wear Failure
- **Features:** 5 sensor readings
- **Accuracy:** ~85%
- **Techniques:** SMOTE for class imbalance, balanced class weights
