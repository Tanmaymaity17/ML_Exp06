# Wine Quality Prediction using Random Forest

This repository contains a **Random Forest model** trained to predict **wine quality** based on physicochemical features. The model is deployed as a **REST API**, allowing users to get predictions by sending JSON data.

---

## 🔗 Deployed API

You can access and test the Wine Quality model API here:  
[https://random-forest-deployed.onrender.com/](https://random-forest-deployed.onrender.com/)

---

## 🧪 Features Used

The model expects the following features in JSON format:

- `Id`  
- `fixed acidity`  
- `volatile acidity`  
- `citric acid`  
- `residual sugar`  
- `chlorides`  
- `free sulfur dioxide`  
- `total sulfur dioxide`  
- `density`  
- `pH`  
- `sulphates`  
- `alcohol`  

---

## ⚡ API Usage

**Endpoint:** `POST /predict`  
**Content-Type:** `application/json`  

**Request Example:**

```json
{
  "features": {
    "Id": 1,
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11,
    "total sulfur dioxide": 34,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }
}
```
**Response Example:**

```json
{
  "prediction": 5
}
```

## ⚡ How to Test the API Locally

You can test the deployed Wine Quality API on any device using this Python script:

```python
import requests

url = "https://random-forest-deployed.onrender.com/predict"
data = {
    "features": {
        "Id": 1,
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11,
        "total sulfur dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
}

response = requests.post(url, json=data)
print(response.json())
