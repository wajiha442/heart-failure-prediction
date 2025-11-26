import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

X = df.drop(columns=["DEATH_EVENT"])
y = df["DEATH_EVENT"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
with open("model/heart_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("Model saved in model/heart_model.pkl")
