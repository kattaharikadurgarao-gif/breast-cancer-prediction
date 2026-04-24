from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

# Load the trained model
try:
    model = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    print("Error: model.pkl not found. Please train and save your model first.")
    model = None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return render_template('index.html', result='Error: Model not loaded'), 500
        
        # Extract form data
        data = [float(x) for x in request.form.values()]
        
        if len(data) == 0:
            return render_template('index.html', result='Error: No features provided'), 400
        
        # Make prediction
        prediction = model.predict([data])
        
        # Convert prediction to readable result
        result = "Malignant" if prediction[0] == 0 else "Benign"
        
        # Support both HTML and JSON responses
        if request.headers.get('Accept') == 'application/json':
            return jsonify({'prediction': result, 'confidence': float(prediction[0])})
        
        return render_template('index.html', result=result)
    
    except ValueError:
        return render_template('index.html', result='Error: Invalid input. Please enter numeric values.'), 400
    except Exception as e:
        return render_template('index.html', result=f'Error: {str(e)}'), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON only)"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid request. Expected features array.'}), 400
        
        features = [float(x) for x in data['features']]
        prediction = model.predict([features])
        result = "Malignant" if prediction[0] == 0 else "Benign"
        
        return jsonify({'prediction': result, 'probability': float(prediction[0])})
    
    except ValueError:
        return jsonify({'error': 'Invalid input. Features must be numeric.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)