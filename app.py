from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model, scaler, and feature names
try:
    model = joblib.load('model/rf_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
except FileNotFoundError:
    print("Error: Model files not found. Please run train_model.py first.")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in the EXACT order of training
        # Features: "FastingGlucose", "HbA1c", "OGTT_2hr", "FastingInsulin", "HOMA_IR",
        # "BMI", "WaistCircumference", "WaistHipRatio", "SystolicBP",
        # "Triglycerides", "HDL", "Age", "FamilyHistory", "PhysicalActivity", "Sex"
        
        input_features = [
            float(data['FastingGlucose']),
            float(data['HbA1c']),
            float(data['OGTT_2hr']),
            float(data['FastingInsulin']),
            float(data['HOMA_IR']),
            float(data['BMI']),
            float(data['WaistCircumference']),
            float(data['WaistHipRatio']),
            float(data['SystolicBP']),
            float(data['Triglycerides']),
            float(data['HDL']),
            float(data['Age']),
            int(data['FamilyHistory']),     # 0 or 1
            float(data['PhysicalActivity']), # Minutes/week? or Index? Assumed numeric.
            int(data['Sex'])                # 0 or 1
        ]
        
        # Reshape for single sample
        final_features = np.array(input_features).reshape(1, -1)
        
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
