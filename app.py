from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load pre-trained models and encoders
crop_model = joblib.load('models/crop_model.joblib')
crop_scaler = joblib.load('models/crop_scaler.joblib')
fert_model = joblib.load('models/fertilizer_model.joblib')
fert_scaler = joblib.load('models/fertilizer_scaler.joblib')
label_encoders = joblib.load('models/fertilizer_label_encoders.joblib')

# Configuration
CROP_PARAMS = {
    'N': (0, 140, 'Nitrogen (ppm)'),
    'P': (5, 145, 'Phosphorous (ppm)'),
    'K': (5, 205, 'Potassium (ppm)'),
    'temperature': (8.8, 43.7, 'Temperature (°C)'),
    'humidity': (15, 99, 'Humidity (%)'),
    'ph': (3.5, 9.9, 'pH Level'),
    'rainfall': (21, 298, 'Rainfall (mm)')
}

FERT_PARAMS = {
    'Temparature': (8, 43, 'Temperature (°C)'),
    'Humidity': (14, 99, 'Humidity (%)'),
    'Moisture': (0.4, 89.0, 'Moisture (%)'),
    'Soil Type': label_encoders['Soil Type'].classes_.tolist(),
    'Crop Type': label_encoders['Crop Type'].classes_.tolist(),
    'Nitrogen': (0, 42, 'Nitrogen (ppm)'),
    'Potassium': (0, 42, 'Potassium (ppm)'),
    'Phosphorous': (5, 42, 'Phosphorous (ppm)')
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        recommendation_type = request.form.get('type')
        
        if recommendation_type == 'crop':
            # Process crop recommendation
            features = [
                float(request.form['N']),
                float(request.form['P']),
                float(request.form['K']),
                float(request.form['temperature']),
                float(request.form['humidity']),
                float(request.form['ph']),
                float(request.form['rainfall'])
            ]
            scaled_features = crop_scaler.transform([features])
            prediction = crop_model.predict(scaled_features)[0]
            accuracy = 0.99  # Replace with actual accuracy
            
            return render_template('result.html', 
                                 recommendation=prediction,
                                 accuracy=accuracy,
                                 type='crop')
        
        elif recommendation_type == 'fertilizer':
            # Process fertilizer recommendation
            soil_type = label_encoders['Soil Type'].transform([request.form['Soil Type']])[0]
            crop_type = label_encoders['Crop Type'].transform([request.form['Crop Type']])[0]
            
            features = [
                float(request.form['Temparature']),
                float(request.form['Humidity']),
                float(request.form['Moisture']),
                soil_type,
                crop_type,
                float(request.form['Nitrogen']),
                float(request.form['Potassium']),
                float(request.form['Phosphorous'])
            ]
            scaled_features = fert_scaler.transform([features])
            prediction = fert_model.predict(scaled_features)[0]
            accuracy = 0.97  # Replace with actual accuracy
            
            return render_template('result.html',
                                 recommendation=prediction,
                                 accuracy=accuracy,
                                 type='fertilizer')
    
    return render_template('recommendation.html',
                         crop_params=CROP_PARAMS,
                         fert_params=FERT_PARAMS)

if __name__ == '__main__':
    app.run(debug=True)