from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and scaler
try:
    model = joblib.load('model/svm_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
except FileNotFoundError:
    print("Error: Model or Scaler not found. Please run train_model.py first.")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in correct order as per training
        # Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        bmi = float(data['bmi'])
        age = float(data['age'])
        glucose = float(data['glucose'])

        features = [
            float(data['pregnancies']),
            glucose,
            float(data['bloodPressure']),
            float(data['skinThickness']),
            float(data['insulin']),
            bmi,
            float(data['diabetesPedigreeFunction']),
            age,
            # Feature Engineering (MUST MATCH TRAIN)
            bmi * age,           # BMI_Age
            glucose * bmi,       # Glucose_BMI
            glucose * age        # Glucose_Age
        ]
        
        # Reshape for single sample
        final_features = np.array(features).reshape(1, -1)
        
        # Scale inputs
        scaled_features = scaler.transform(final_features)
        
        # Predict
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)[0][1]
        
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        
        return jsonify({
            'prediction': result,
            'probability': round(probability * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
