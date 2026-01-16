from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.txt")

# Load model
model = joblib.load(MODEL_PATH)

# Load class names
with open(CLASS_PATH) as f:
    class_names = [line.strip() for line in f]

# Feature order for prediction
features = [
    "age","trestbps","chol","thalach","oldpeak",
    "sex","cp","thal","restecg","slope",
    "exang","ca","fbs"
]

@app.route("/")
def index():
    return render_template("index_25RP21648.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data], columns=features)

        pred_class = int(model.predict(df)[0])
        probs = model.predict_proba(df)[0]

        # Map probabilities in the desired order
        result = {
            "Actual Class": class_names[pred_class],
            "Prob_immediate danger": float(probs[4]),
            "Prob_mild": float(probs[2]),
            "Prob_no disease": float(probs[0]),
            "Prob_severe": float(probs[3]),
            "Prob_very mild": float(probs[1]),
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
