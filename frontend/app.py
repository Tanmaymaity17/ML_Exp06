import streamlit as st
import requests

st.set_page_config(page_title="Wine Quality Predictor", layout="centered")
st.title("🍷 Wine Quality Predictor")

API_URL = "https://random-forest-deployed.onrender.com/predict"  # Your Render API URL

st.write("Enter the wine features below:")

features = {
    "Id": 1,
    "fixed acidity": st.slider("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.4, step=0.1),
    "volatile acidity": st.slider("Volatile Acidity", min_value=0.0, max_value=5.0, value=0.7, step=0.01),
    "citric acid": st.slider("Citric Acid", min_value=0.0, max_value=5.0, value=0.0, step=0.01),
    "residual sugar": st.slider("Residual Sugar", min_value=0.0, max_value=20.0, value=1.9, step=0.1),
    "chlorides": st.slider("Chlorides", min_value=0.0, max_value=1.0, value=0.076, step=0.001),
    "free sulfur dioxide": st.slider("Free Sulfur Dioxide", min_value=0, max_value=100, value=11, step=1),
    "total sulfur dioxide": st.slider("Total Sulfur Dioxide", min_value=0, max_value=300, value=34, step=1),
    "density": st.slider("Density", min_value=0.9, max_value=1.1, value=0.9978, step=0.0001),
    "pH": st.slider("pH", min_value=0.0, max_value=14.0, value=3.51, step=0.01),
    "sulphates": st.slider("Sulphates", min_value=0.0, max_value=5.0, value=0.56, step=0.01),
    "alcohol": st.slider("Alcohol", min_value=0.0, max_value=20.0, value=9.4, step=0.1)
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
