from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and preprocessing objects
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()

        # Convert to proper types
        pclass = int(data['pclass'])
        sex = data['sex']
        age = float(data['age'])
        sibsp = int(data['sibsp'])
        parch = int(data['parch'])
        fare = float(data['fare'])
        embarked = data['embarked']

        # Preprocess
        sex_encoded = label_encoder.transform([sex])[0]
        embarked_encoded = label_encoder.transform([embarked])[0]

        # Calculate derived features
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0

        # Scale numerical features
        age_fare_scaled = scaler.transform([[age, fare]])
        age_scaled = age_fare_scaled[0][0]
        fare_scaled = age_fare_scaled[0][1]

        # Create feature array
        features = [pclass, sex_encoded, age_scaled, sibsp, parch, fare_scaled,
                   embarked_encoded, family_size, is_alone]

        # Make prediction
        prediction = model.predict([features])
        probability = model.predict_proba([features])[0][1]

        # Return result
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'message': 'Survived' if prediction[0] == 1 else 'Did not survive'
        }

        return render_template('index.html', prediction_text=f"Prediction: {result['message']} (Probability: {result['probability']:.2f})")

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
