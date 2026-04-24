import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])

    result = "Malignant" if prediction[0] == 0 else "Benign"
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run()
  
return render_template('index.html')