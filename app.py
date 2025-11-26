import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler 

 
try:
    with open("heart_failure_svm_model.pkl", "rb") as f:
        model, scaler = pickle.load(f) 
    st.success("Model aur Scaler safalta poorvak load ho chuke hain!")

except FileNotFoundError:
    st.error("Model file nahi mili ya GitHub par push nahi hui. Filename aur jagah (root folder) check karein.")
    st.stop()
except ValueError:
    st.error("Model file theek se load nahi hui (cannot unpack). Confirm karein ki training mein model aur scaler dono ek saath save kiye gaye hain.")
    st.stop()
except Exception as e:
    st.error(f"Model load karne mein koi aur masla aaya: {e}")
    st.stop()

st.title("❤️ Heart Failure Prediction App")
st.write("Please zaroori information fill karein takay hum Heart Failure ka risk predict kar sakein. Sabhi 12 features zaroori hain.")

# Input fields for all 12 raw features
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("1. Umar (Age)", 40, 95, 60)
    ejection_fraction = st.slider("2. Ejection Fraction (%)", 10, 80, 40)
    serum_sodium = st.slider("3. Serum Sodium (mEq/L)", 113, 148, 137)
    time = st.number_input("4. Follow-up period (Days)", min_value=10, max_value=300, value=150)

with col2:
    creatinine_phosphokinase = st.number_input("5. CPK (Creatinine Phosphokinase)", value=582, min_value=23)
    platelets = st.number_input("6. Platelets (kiloplatelets/mL)", value=263.35, min_value=25.0)
    serum_creatinine = st.number_input("7. Serum Creatinine (mg/dL)", value=1.0, min_value=0.5, max_value=10.0)
    sex = st.selectbox("8. Jins (Sex)", options=[('Male', 1), ('Female', 0)], format_func=lambda x: x[0])

with col3:
    anaemia = st.selectbox("9. Anaemia", options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])
    diabetes = st.selectbox("10. Diabetes", options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])
    high_blood_pressure = st.selectbox("11. High Blood Pressure", options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])
    smoking = st.selectbox("12. Smoking", options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])


if st.button("Predict Heart Failure"):
    
     
    age_group_Middle = 1 if 50 < age <= 70 else 0
    age_group_Old = 1 if age > 70 else 0
    
    
    ejection_fraction_group_Low = 1 if ejection_fraction <= 30 else 0
    ejection_fraction_group_High = 1 if ejection_fraction > 50 else 0
    
    
    serum_sodium_group_Normal = 1 if 135 < serum_sodium <= 145 else 0
    serum_sodium_group_High = 1 if serum_sodium > 145 else 0

     
    raw_data_preprocessed = np.array([[
        age, 
        anaemia[1], 
        creatinine_phosphokinase, 
        diabetes[1], 
        ejection_fraction, 
        high_blood_pressure[1], 
        platelets, 
        serum_creatinine, 
        serum_sodium, 
        smoking[1], 
        time,
        sex[1],
        
        
        age_group_Middle, 
        age_group_Old,
        ejection_fraction_group_Low,
        ejection_fraction_group_High,
        serum_sodium_group_Normal,
        serum_sodium_group_High
    ]])

    try:
        scaled_data = scaler.transform(raw_data_preprocessed)
    except ValueError as e:
        st.error(f"Scaling Error: Data mein {raw_data_preprocessed.shape[1]} features hain, lekin Scaler ko {scaler.n_features_in_} features chahiye. Training aur App features ka count/order check karein. {e}")
        st.stop()
    except Exception as e:
        st.error(f"Data scale karne mein masla: {e}")
        st.stop()

    prediction = model.predict(scaled_data)
    
    try:
        prediction_proba = model.predict_proba(scaled_data)[:, 1]
    except AttributeError:
        prediction_proba = [0.0 if prediction[0] == 0 else 1.0]

    st.subheader("Prediction Result:")
    
    if prediction[0] == 1:
        st.error(f"High Risk of Heart Failure! (Probability: {prediction_proba[0]*100:.2f}%)")
    else:
        st.success(f"Low Risk of Heart Failure. (Probability: {prediction_proba[0]*100:.2f}%)")
        
    st.info("⚠️ Yeh sirf ek machine learning prediction hai. Hamesha expert medical advice lein.")
