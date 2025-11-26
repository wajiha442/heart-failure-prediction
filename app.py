import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

try:
    with open("heart_model.pkl", "rb") as f:
        model, scaler = pickle.load(f)
    st.success("Model aur Scaler safalta poorvak load ho chuke hain!")

except FileNotFoundError:
    st.error("Model file nahi mili (heart_model.pkl). Please confirm karein ki file root folder mein hai.")
    st.stop()
except Exception as e:
    st.error(f"Model load karne mein koi aur masla aaya: {e}")
    st.stop()

st.title("❤️ Heart Failure Prediction App")
st.write("Please zaroori information fill karein takay hum Heart Failure ka risk predict kar sakein.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Umar (Age)", 18, 100, 50)
    ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 40)
    platelets = st.number_input("Platelets (kiloplatelets/mL)", value=250.0)

with col2:
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", value=1.0)
    serum_sodium = st.slider("Serum Sodium (mEq/L)", 110, 150, 137)
    time = st.number_input("Follow-up period (Days)", value=100)
    
sex = st.selectbox("Jins (Sex)", options=[('Male', 1), ('Female', 0)], format_func=lambda x: x[0])
anaemia = st.selectbox("Anaemia", options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])

if st.button("Predict Heart Failure"):
    
    raw_data = np.array([[
        age, 
        anaemia[1],
        ejection_fraction, 
        platelets, 
        serum_creatinine, 
        serum_sodium, 
        sex[1],
        time
    ]])

    try:
        scaled_data = scaler.transform(raw_data)
    except Exception as e:
        st.error(f"Data scale karne mein masla (Scaler Error): {e}")
        st.stop()

    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)[:, 1]

    st.subheader("Prediction Result:")
    
    if prediction[0] == 1:
        st.error(f"High Risk of Heart Failure! (Probability: {prediction_proba[0]*100:.2f}%)")
    else:
        st.success(f"Low Risk of Heart Failure. (Probability: {prediction_proba[0]*100:.2f}%)")
        
    st.info("⚠️ Yeh sirf ek machine learning prediction hai. Hamesha expert medical advice lein.")

