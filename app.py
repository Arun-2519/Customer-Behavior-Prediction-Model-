import streamlit as st
import pickle
import numpy as np
from pathlib import Path
import pandas as pd

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent

model = pickle.load(open(BASE_DIR / "customer_model.pkl", "rb"))
scaler = pickle.load(open(BASE_DIR / "customer_scaler.pkl", "rb"))

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Customer Behavior Predictor", layout="centered")

st.title(" Customer Behavior Prediction System")

st.markdown("Predict customer value based on their profile and purchase behavior.")

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("📥 Enter Customer Details")

age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)

gender = st.selectbox("Gender", ["Male", "Female"])

total_spent = st.number_input("Total Amount Spent (₹)", min_value=0.0, value=1000.0)

order_count = st.number_input("Number of Orders", min_value=0, value=5)

avg_order = st.number_input("Average Order Value (₹)", min_value=0.0, value=200.0)

# Encode gender
gender_val = 1 if gender == "Male" else 0

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Customer Value"):

    input_data = np.array([[age, gender_val, total_spent, order_count, avg_order]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    # -------------------------------
    # RESULT INTERPRETATION
    # -------------------------------
    st.subheader(" Prediction Result")

    st.success(f"💰 Predicted Customer Value: ₹{prediction:.2f}")

    # Simple classification
    if prediction < 1000:
        st.warning("Low Value Customer")
    elif prediction < 5000:
        st.info(" Medium Value Customer")
    else:
        st.success(" High Value Customer")

    # -------------------------------
    # SUMMARY TABLE
    # -------------------------------
    st.subheader("📋 Summary")

    summary = {
        "Age": age,
        "Gender": gender,
        "Total Spent": total_spent,
        "Order Count": order_count,
        "Avg Order Value": avg_order,
        "Predicted Value": round(prediction, 2)
    }

    st.table(pd.DataFrame(summary.items(), columns=["Metric", "Value"]))

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("💡 Helps businesses understand customer value and behavior")
