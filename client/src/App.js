import React, { useState } from 'react';

function App() {
  const [formData, setFormData] = useState({
    type: 'L',
    air_temperature: '',
    process_temperature: '',
    rotational_speed: '',
    torque: '',
    tool_wear: ''
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          type: formData.type,
          air_temperature: parseFloat(formData.air_temperature),
          process_temperature: parseFloat(formData.process_temperature),
          rotational_speed: parseFloat(formData.rotational_speed),
          torque: parseFloat(formData.torque),
          tool_wear: parseFloat(formData.tool_wear)
        })
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError('Failed to connect to server. Make sure the Flask backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const getFailureTypeColor = (prediction) => {
    return prediction === 'No Failure' ? 'no-failure' : 'failure';
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Predictive Maintenance</h1>
        <p>Failure Type Prediction using Machine Learning</p>
      </header>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="type">Product Quality Type</label>
            <select
              id="type"
              name="type"
              value={formData.type}
              onChange={handleChange}
              required
            >
              <option value="L">Low (L)</option>
              <option value="M">Medium (M)</option>
              <option value="H">High (H)</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="air_temperature">Air Temperature (K)</label>
            <input
              type="number"
              id="air_temperature"
              name="air_temperature"
              placeholder="e.g., 298.1"
              step="0.1"
              value={formData.air_temperature}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="process_temperature">Process Temperature (K)</label>
            <input
              type="number"
              id="process_temperature"
              name="process_temperature"
              placeholder="e.g., 308.6"
              step="0.1"
              value={formData.process_temperature}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="rotational_speed">Rotational Speed (rpm)</label>
            <input
              type="number"
              id="rotational_speed"
              name="rotational_speed"
              placeholder="e.g., 1551"
              step="1"
              value={formData.rotational_speed}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="torque">Torque (Nm)</label>
            <input
              type="number"
              id="torque"
              name="torque"
              placeholder="e.g., 42.8"
              step="0.1"
              value={formData.torque}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="tool_wear">Tool Wear (min)</label>
            <input
              type="number"
              id="tool_wear"
              name="tool_wear"
              placeholder="e.g., 0"
              step="1"
              value={formData.tool_wear}
              onChange={handleChange}
              required
            />
          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? 'Predicting...' : 'Predict Failure Type'}
          </button>
        </form>

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Analyzing sensor data...</p>
          </div>
        )}

        {error && (
          <div className="result error">
            <h3>Error</h3>
            <p>{error}</p>
          </div>
        )}

        {result && (
          <div className="result success">
            <h3>Predicted Failure Type</h3>
            <div className={`prediction-type ${getFailureTypeColor(result.prediction)}`}>
              {result.prediction}
            </div>
            <p className="confidence">Confidence: {(result.confidence * 100).toFixed(1)}%</p>

            <div className="probabilities">
              {Object.entries(result.all_probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([type, prob]) => (
                  <div key={type} className="prob-bar">
                    <span className="prob-label">{type}</span>
                    <div className="prob-track">
                      <div
                        className="prob-fill"
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
