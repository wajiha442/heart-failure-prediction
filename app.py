 import streamlit as st
import numpy as np
import pickle

with open("model/heart_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

st.title("Heart Failure Prediction App")

age = st.number_input("Age", 1, 120, 50)
anaemia = st.selectbox("Anaemia", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", 0, 5000, 100)
diabetes = st.selectbox("Diabetes", [0, 1])
ejection_fraction = st.number_input("Ejection Fraction", 1, 100, 38)
high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
platelets = st.number_input("Platelets", 10000, 1000000, 250000)
serum_creatinine = st.number_input("Serum Creatinine", 0.0, 10.0, 1.2)
serum_sodium = st.number_input("Serum Sodium", 100, 200, 135)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
smoking = st.selectbox("Smoking", [0, 1])
time = st.number_input("Follow-up Period", 1, 500, 150)

if st.button("Predict"):
    data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes,
                      ejection_fraction, high_blood_pressure, platelets,
                      serum_creatinine, serum_sodium, sex, smoking, time]])
    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]

    if pred == 1:
        st.error("⚠️ High Risk of Heart Failure")
    else:
        st.success("✅ Low Risk of Heart Failure")
