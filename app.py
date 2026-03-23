import streamlit as st
import pandas as pd
import joblib
from keras.models import load_model

# Load model and scaler
model = load_model("insurance_model.keras")
scaler = joblib.load("scaler.pkl")

st.title("Medical Insurance Price Prediction")
st.write("Enter details below to predict insurance cost")

# User Inputs
age = st.number_input("Age", 18, 100)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0)
children = st.number_input("Children", 0, 5)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict Insurance Cost"):

    # Create DataFrame
    data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Encode binary columns
    data["sex"] = data["sex"].map({"male": 0, "female": 1})
    data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})

    # One-hot encode region
    data = pd.get_dummies(data, columns=["region"])

    # 🔥 FIX: include ALL region columns (including northeast)
    data = data.reindex(columns=[
    'age','sex','bmi','children','smoker',
    'region_northeast','region_northwest',
    'region_southeast','region_southwest'
], fill_value=0)

    # Scale input
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)

    st.success(f"Predicted Insurance Cost: {prediction[0][0]:.2f}")
