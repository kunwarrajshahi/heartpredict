from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("heart_disease_model.pkl")

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    features = np.array([[data[f] for f in FEATURE_NAMES]])

    prediction = int(model.predict(features)[0])
    probability = model.predict_proba(features)[0]

    # Feature importance (Random Forest)
    importance = model.feature_importances_

    return jsonify({
        "prediction": prediction,
        "result": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
        "probability": {
            "no_disease": round(probability[0] * 100, 2),
            "disease": round(probability[1] * 100, 2)
        },
        "feature_importance": dict(
            zip(FEATURE_NAMES, importance.round(4))
        )
    })

if __name__ == "__main__":
    app.run(debug=True)
