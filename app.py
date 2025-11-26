import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from IPython.display import display

print("Loading Heart Failure Dataset...")

possible_paths = [
    "/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv",
    "/kaggle/input/heart-failure/heart.csv",
    "heart.csv",
    "heart_failure_clinical_records_dataset.csv"
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded dataset from: {path}")
        break

if df is None:
    raise FileNotFoundError("Heart Failure dataset not found.")

print(f"Dataset Shape: {df.shape}")
display(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

if "DEATH_EVENT" not in df.columns:
    raise ValueError("Target column DEATH_EVENT missing.")

print("\nApplying Binning on Key Numerical Columns...")

df["age_group"] = pd.cut(df["age"], bins=[0, 50, 70, 100], labels=["Young", "Middle", "Old"])
df["ejection_fraction_group"] = pd.cut(df["ejection_fraction"], bins=[0, 30, 50, 100], labels=["Low", "Medium", "High"])
df["serum_sodium_group"] = pd.cut(df["serum_sodium"], bins=[0, 135, 145, 200], labels=["Low", "Normal", "High"])

display(df.head())

print("\nPlotting Binned Features...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(x='age_group', data=df, ax=axes[0])
sns.countplot(x='ejection_fraction_group', data=df, ax=axes[1])
sns.countplot(x='serum_sodium_group', data=df, ax=axes[2])
axes[0].set_title("Age Categories")
axes[1].set_title("Ejection Fraction Categories")
axes[2].set_title("Serum Sodium Categories")
plt.tight_layout()
plt.show()

print("\nOne-Hot Encoding...")

df_encoded = pd.get_dummies(df,
                            columns=["age_group", "ejection_fraction_group", "serum_sodium_group"],
                            drop_first=True)

display(df_encoded.head())

X = df_encoded.drop(columns=["DEATH_EVENT"])
y = df_encoded["DEATH_EVENT"]

print("\nSplitting Train/Test Sets...")
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train Shape = {x_train.shape}")
print(f"Test Shape = {x_test.shape}")

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("\nTraining SVM Model...")

model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("\nAccuracy Score:")
acc = accuracy_score(y_test, y_pred)
print(f"{acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Heart Failure — Confusion Matrix")
plt.show()

# Save the fitted scaler
scaler_filename = "scaler.pkl"
with open(scaler_filename, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler saved as: {scaler_filename}")

# Save the model
model_filename = "heart_failure_svm_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved as: {model_filename}")
