import streamlit as st
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "behavior_model.pkl")
scaler = joblib.load(BASE_DIR / "behavior_scaler.pkl")

st.title("🧠 Customer Behavior Prediction")

customer_id = st.text_input("Customer ID", "C0001")
age = st.number_input("Age", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
total_spent = st.number_input("Total Spent", 0.0, 100000.0, 1000.0)
order_count = st.number_input("Order Count", 0, 1000, 5)
avg_order = st.number_input("Avg Order Value", 0.0, 10000.0, 200.0)
std_dev = st.number_input("Spending Variation (Std Dev)", 0.0, 10000.0, 50.0)

gender_val = 1 if gender == "Male" else 0

if st.button("Predict Behavior"):

    input_data = np.array([[age, gender_val, total_spent, order_count, avg_order, std_dev]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        behavior = "Predictable Buyer"
        reason = "Very consistent spending"
    elif prediction == 1:
        behavior = "Moderate Buyer"
        reason = "Moderately consistent spending"
    else:
        behavior = "Irregular Buyer"
        reason = "Spending varies a lot"

    st.success(f"""
Customer {customer_id}:
Behavior: {behavior}
Reason: {reason}
""")
