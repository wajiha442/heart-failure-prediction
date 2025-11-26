# Heart Failure Prediction Web App

This project is a Machine Learning–powered Heart Failure Prediction App built using Streamlit.

Users enter medical data, and the app predicts whether the person is at high risk or low risk of heart failure.

## Features

- Streamlit web interface
- Logistic Regression ML model
- Scaled numerical features
- Easy deployment

## Project Structure

project-folder/
│── app.py
│── train_model.py
│── heart_failure_clinical_records_dataset.csv
│── model/
│ └── heart_model.pkl
│ └── scaler.pkl
│── requirements.txt
│── README.md

## How to Run

### Install requirements

pip install -r requirements.txt

### Train the model

python train_model.py

### Run the app

streamlit run app.py

## Output

The app predicts:

- High Risk of Heart Failure ⚠️
- Low Risk of Heart Failure ✅
