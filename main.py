import requests

url = "https://random-forest-deployed.onrender.com/predict"
data = {
    "features": {
        "Id": 1,  # can be any number or string
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
