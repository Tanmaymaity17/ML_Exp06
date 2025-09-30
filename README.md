# Deploying a Random Forest Model - Wine Quality Prediction — Cookbook

A step‑by‑step, easy-to-follow cookbook to reproduce the full project: training a Random Forest on the Wine Quality dataset, packaging the trained model behind a Flask REST API, deploying to Render, and testing the deployed API.

**Repo & deployed API (for reference)**

- GitHub repo: `https://github.com/Tanmaymaity17/ML_Exp06`
- Deployed Frontend: `https://random-forest-tanmay.streamlit.app/`
- Deployed API: `https://random-forest-deployed.onrender.com/`
- Dataset: `https://www.kaggle.com/datasets/yasserh/wine-quality-dataset`

---

# 1. Goal & Overview

**Goal:** Train a Random Forest classifier to predict wine quality, expose it with a Flask API, deploy it to Render, and provide simple test clients.

**High-level steps:**

1. Prepare environment & data
2. Train the Random Forest model and tune parameters
3. Save the trained model
4. Build a Flask API that loads the model and returns predictions
5. Deploy to Render
6. Test the API (Python, Postman)

---

# 2. Prerequisites

- Python 3.8+ installed
- Git installed and a GitHub account
- Render account if you want to deploy
- A copy of the Wine Quality dataset (`WineQT.csv`) — download from Kaggle if needed

---

# 3. Project structure

```
ML_Exp06/
├── train_model.py         # training + evaluation script
├── app.py                 # Flask API
├── wine_quality_rf.pkl    # saved model (after training)
├── WineQT.csv             # dataset (or exclude from repo and load locally)
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 4. Setup dependencies

requirements.txt :

```
flask
scikit-learn
pandas
numpy
joblib
gunicorn
```
Run :

```powershell
pip install -r requirements.txt
```

---

# 5. Data loading & Train the Random Forest

### About Random Forest (short)

Random Forest is an ensemble of decision trees trained with bootstrap aggregation (bagging). Each tree sees a random subset of the samples and a random subset of features; final predictions are made by majority vote (classification).

Key parameters (short):
- `n_estimators` — number of trees (more trees → more stable predictions, slower training).
- `criterion` — split metric (`'gini'` or `'entropy'`) that affects how splits are chosen.
- `max_features` — number of features considered at each split (`'sqrt'`, `'log2'`, integer, or `None`).
- `max_depth` — maximum depth of each tree (limits overfitting).
- `min_samples_split` / `min_samples_leaf` — regularization to avoid tiny leaf nodes.
- `bootstrap` — whether to use bootstrap samples (default `True`).
- `random_state` — fixed seed for reproducibility.

Why used here: robust for tabular data, requires little preprocessing, handles nonlinear feature interactions, and gives good baseline performance quickly.

- Random Forest Classifier Documentation : `https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html`

---

### Create `train_model.py`:

```python
from math import log2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset from local file
data = pd.read_csv("WineQT.csv")

# Inspect column names
print("Columns:", data.columns)

# Features and target (assuming target is 'quality')
X = data.drop("quality", axis=1)
y = data["quality"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RandomForestClassifier
model = RandomForestClassifier(criterion="entropy", max_features="log2", random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save trained model
joblib.dump(model, "wine_quality_rf.pkl")
print("Model saved as wine_quality_rf.pkl")
```
---
### Hyperparameter experiments

We tried several Random Forest settings and recorded quick observations:

- `n_estimators`: tried 100 and 200 — more trees gave only marginal gains while increasing training time.
- `criterion`: compared `'gini'` vs `'entropy'` — `'entropy'` produced better results for this dataset.
- `max_features`: tried `'sqrt'`, `'log2'`, and `None` — `'log2'` yielded the best validation accuracy.
- `max_depth`, `min_samples_split`, `min_samples_leaf`: tuned but changes did not further increase accuracy in a meaningful way; these mostly affected overfitting/underfitting trade-offs.
- `min_weight_fraction_leaf`, `bootstrap`: tested defaults — no significant improvement noted.

Baseline (default settings) accuracy ≈ **68%** → After tuning (final params `criterion='entropy', max_features='log2', random_state=42`) accuracy improved to **70.74%**.

---

# 6. Build the Flask API

Create `app.py` :

```python
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
```

**Important notes:**

- Make sure column order and names align with the model (training order). 
- Use `os.environ.get('PORT')` so Render (or other hosts) can provide the port.
- Add `flask-cors` if you plan to call the API from a browser.

---

# 7. Add Streamlit Frontend

Create a new folder `/frontend` in your repo:
```python
ML_Exp06/
├── frontend/
│   ├── app.py
│   └── requirements.txt
```
`frontend/app.py` :
```python
import streamlit as st
import requests

st.set_page_config(page_title="Wine Quality Predictor", layout="centered")
st.title("🍷 Wine Quality Predictor")

API_URL = "https://random-forest-deployed.onrender.com/predict"  # Deployed Flask API

st.write("Enter the wine features below using sliders:")

features = {
    "Id": st.slider("ID", min_value=1, max_value=1000, value=1, step=1),
    "fixed acidity": st.slider("Fixed Acidity", 0.0, 20.0, 7.4, 0.1),
    "volatile acidity": st.slider("Volatile Acidity", 0.0, 5.0, 0.7, 0.01),
    "citric acid": st.slider("Citric Acid", 0.0, 5.0, 0.0, 0.01),
    "residual sugar": st.slider("Residual Sugar", 0.0, 20.0, 1.9, 0.1),
    "chlorides": st.slider("Chlorides", 0.0, 1.0, 0.076, 0.001),
    "free sulfur dioxide": st.slider("Free Sulfur Dioxide", 0, 100, 11, 1),
    "total sulfur dioxide": st.slider("Total Sulfur Dioxide", 0, 300, 34, 1),
    "density": st.slider("Density", 0.9, 1.1, 0.9978, 0.0001),
    "pH": st.slider("pH", 0.0, 14.0, 3.51, 0.01),
    "sulphates": st.slider("Sulphates", 0.0, 5.0, 0.56, 0.01),
    "alcohol": st.slider("Alcohol", 0.0, 20.0, 9.4, 0.1)
}

if st.button("Predict Quality"):
    try:
        response = requests.post(API_URL, json={"features": features})
        result = response.json()
        if "prediction" in result:
            st.success(f"Predicted Wine Quality: {result['prediction']}")
        else:
            st.error(f"API Error: {result.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Request failed: {e}")

```

---

# 8. Testing

### 1) Streamlit Frontend :
 - Adjust the wine features using sliders.

- Click Predict Quality to see results fetched from the deployed Flask API.
- Deployed frontend : `https://random-forest-tanmay.streamlit.app/`

### 2) Python `requests`:

**Note:** Run the script below on any device with internet access and Python installed. Change the feature values as required — the prediction computation runs on the deployed backend at Render, so you don't need the model or training dependencies locally.


```python
import requests

url = 'https://random-forest-deployed.onrender.com/predict'
json_data = {
    'features': {
        'Id': 1,
        'fixed acidity': 7.4,
        'volatile acidity': 0.7,
        'citric acid': 0,
        'residual sugar': 1.9,
        'chlorides': 0.076,
        'free sulfur dioxide': 11,
        'total sulfur dioxide': 34,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4
    }
}

r = requests.post(url, json=json_data)
print(r.status_code, r.json())
```


### 3) Postman:

- Create a POST request to `https://random-forest-deployed.onrender.com/predict`
- Select Body → raw → JSON and paste the JSON from above
- Send → view JSON response

---

# 9. Common errors & troubleshooting

**Error:** `Missing feature: fixed acidity` or similar

- Cause: your JSON used `fixed_acidity` (underscore) while the API expects `fixed acidity` (space).
- Fix: send keys with exact names expected by the API.

**Error:** `Missing feature: Id`

- Cause: the model validation expects an `Id` column.
- Fix: include an `Id` field in the JSON body (any int/string is fine).

**Git push failing with "Password authentication is not supported"**

- Solution: use a GitHub Personal Access Token (PAT) instead of your password or set up SSH keys.

---

# 10. Deployment Render & Streamlit — quick steps

**10.1 Deploy Flask API :**

1. Commit your code (including `main.py`, `requirements.txt`, and `wine_quality_rf.pkl`). Push to GitHub.
2. Create a Render account and connect your GitHub repo.
3. Create a new **Web Service** → choose your repo and branch (`main`).
4. Set **Build Command**: `pip install -r requirements.txt`
5. Set **Start Command**: `gunicorn main:app --bind 0.0.0.0:$PORT`.
6. Deploy. When done, your app will be available at a Render URL (use your deployed URL).

**10.2 Deploy Streamlit frontend :**

1. Commit /frontend folder to GitHub
2. Go to Streamlit Cloud → New app
3. Connect GitHub repo → select frontend folder
4. Main file: app.py
5. Deploy → get Streamlit frontend URL

**Tips:**

- If model file is large, consider storing it in cloud object storage and load at runtime — or keep it under LFS / external bucket.
- Make sure `wine_quality_rf.pkl` path used in `app.py` matches the file location in the repo.

---
# Conclusion

This project covered the full lifecycle: dataset preparation (Wine Quality), model training and light hyperparameter tuning, saving the trained Random Forest model, wrapping it in a Flask REST API, deploying to Render, and testing using Python/Postman. Through targeted tuning we improved baseline accuracy from **68%** to **70.74%** using `criterion='entropy'` and `max_features='log2'`. The deployed API enables predictions without local model files.
Users can now adjust wine features interactively via sliders and instantly get predictions from the deployed backend, making the system fully accessible and user-friendly.

---
