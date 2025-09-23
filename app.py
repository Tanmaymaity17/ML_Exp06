# app.py
import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load model & feature names
MODEL_PATH = "wine_quality_rf.pkl"
FEATURES_PATH = "feature_names.pkl"
CSV_PATH = "WineQT.csv"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Run train_model.py first.")

model = joblib.load(MODEL_PATH)

if os.path.exists(FEATURES_PATH):
    feature_names = joblib.load(FEATURES_PATH)
else:
    # fallback: read CSV to infer features
    feature_names = pd.read_csv(CSV_PATH).drop("quality", axis=1).columns.tolist()

app = Flask(__name__)

@app.route("/")
def index():
    return {"status": "ok", "message": "Wine Quality model API. POST to /predict with JSON {'features': {..}}."}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    features = data.get("features")
    if features is None:
        return jsonify({"error": "JSON must contain key 'features' (dict or list).", "expected_features": feature_names}), 400

    # Accept dict (named features) or list (ordered)
    if isinstance(features, dict):
        try:
            values = [features[name] for name in feature_names]
        except KeyError as e:
            missing = str(e).strip("'")
            return jsonify({"error": f"Missing feature: {missing}", "expected_features": feature_names}), 400
    elif isinstance(features, list):
        values = features
        if len(values) != len(feature_names):
            return jsonify({"error": "Feature length mismatch", "expected_length": len(feature_names)}), 400
    else:
        return jsonify({"error": "features must be a dict or list"}), 400

    arr = np.array(values).reshape(1, -1)
    pred = model.predict(arr)[0]
    return jsonify({"prediction": int(pred)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
