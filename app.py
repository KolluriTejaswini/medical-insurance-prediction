import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model
model = load_model("insurance_model.keras")

# Load scaler
scaler = joblib.load("scaler.pkl")

st.title("Medical Insurance Price Prediction")

age = st.number_input("Age",18,100)
sex = st.selectbox("Sex",["male","female"])
bmi = st.number_input("BMI",10.0,50.0)
children = st.number_input("Children",0,5)
smoker = st.selectbox("Smoker",["yes","no"])
region = st.selectbox("Region",
["northeast","northwest","southeast","southwest"])

if st.button("Predict"):

    data = pd.DataFrame({
        "age":[age],
        "sex":[sex],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker],
        "region":[region]
    })

    data["sex"] = data["sex"].map({"male":0,"female":1})
    data["smoker"] = data["smoker"].map({"no":0,"yes":1})

    data = pd.get_dummies(data, columns=["region"])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    st.success(f"Predicted Insurance Cost: {prediction[0][0]}")
