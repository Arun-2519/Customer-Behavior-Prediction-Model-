import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "customer_model.pkl")
scaler = joblib.load(BASE_DIR / "customer_scaler.pkl")

# -------------------------------
# UI
# -------------------------------
st.title("🧠 Customer Prediction App")

st.write("Enter customer details to predict value.")

# -------------------------------
# INPUTS (MUST MATCH TRAINING)
# -------------------------------
age = st.number_input("Age", 18, 100, 30)

gender = st.selectbox("Gender", ["Male", "Female"])

total_spent = st.number_input("Total Spent", 0.0, 100000.0, 1000.0)

order_count = st.number_input("Order Count", 0, 1000, 5)

avg_order = st.number_input("Average Order Value", 0.0, 10000.0, 200.0)

# Encode gender
gender_val = 1 if gender == "Male" else 0

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):

    input_data = np.array([[age, gender_val, total_spent, order_count, avg_order]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Value: {prediction:.2f}")
