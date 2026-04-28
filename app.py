import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("📊 Customer Behavior Prediction (Clean & Stable)")

# ------------------ UPLOAD ------------------
customers_file = st.file_uploader("Upload Customers Dataset", type=["csv"])
orders_file = st.file_uploader("Upload Orders Dataset", type=["csv"])

if customers_file and orders_file:

    customers = pd.read_csv(customers_file)
    orders = pd.read_csv(orders_file)

    st.success("Datasets Loaded ✅")

    # ------------------ CLEAN CUSTOMER ID ------------------
    st.header("🧹 Data Cleaning")

    # Drop rows where Customer ID is missing
    customers = customers.dropna(subset=["Customer ID"])
    orders = orders.dropna(subset=["Customer ID"])

    # Convert to string (important)
    customers["Customer ID"] = customers["Customer ID"].astype(str)
    orders["Customer ID"] = orders["Customer ID"].astype(str)

    st.write("Customers Cleaned", customers.head())
    st.write("Orders Cleaned", orders.head())

    # ------------------ AGGREGATE ORDERS ------------------
    st.header("📊 Orders Aggregation")

    # Example aggregation
    orders_agg = orders.groupby("Customer ID").agg({
        "Purchase Amount": "sum"
    }).reset_index()

    st.write("Aggregated Orders", orders_agg.head())

    # ------------------ MERGE ------------------
    st.header("🔗 Merge Data")

    df = pd.merge(customers, orders_agg, on="Customer ID", how="inner")

    if len(df) == 0:
        st.error("❌ Merge failed — no matching Customer IDs")
        st.stop()

    st.success("Merged Successfully ✅")
    st.write(df.head())

    # ------------------ FEATURE ENGINEERING ------------------
    st.header("⚙️ Feature Engineering")

    # Create spending category
    df["Spending Level"] = pd.cut(
        df["Purchase Amount"],
        bins=[0, 500, 2000, 10000],
        labels=["Low", "Medium", "High"]
    )

    st.write(df.head())

    # ------------------ TARGET ------------------
    target_col = st.selectbox("Select Target Column", df.columns)

    # ------------------ TRAIN ------------------
    if st.button("🚀 Train Model"):

        try:
            df = df.dropna()

            X = df.drop(columns=[target_col])
            y = df[target_col]
            customer_ids = df["Customer ID"]

            # Convert categorical
            X = pd.get_dummies(X)

            # Encode target
            y_encoded, labels = pd.factorize(y)

            # Split
            X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
                X, y_encoded, customer_ids, test_size=0.2, random_state=42
            )

            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # ------------------ MODEL ------------------
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # ------------------ METRICS ------------------
            train_acc = model.score(X_train, y_train)
            test_acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            st.subheader("📊 Model Performance")
            st.write(f"Train Accuracy: {train_acc:.3f}")
            st.write(f"Test Accuracy: {test_acc:.3f}")
            st.write(f"Precision: {prec:.3f}")
            st.write(f"Recall: {rec:.3f}")
            st.write(f"F1 Score: {f1:.3f}")

            # ------------------ OUTPUT ------------------
            st.subheader("📦 Customer Behavior Output")

            for i in range(len(y_pred)):

                customer_id = id_test.iloc[i]
                behavior = labels[y_pred[i]]

                purchase = df.iloc[id_test.index[i]]["Purchase Amount"]

                if purchase > df["Purchase Amount"].mean():
                    reason = "Very consistent spending"
                else:
                    reason = "Irregular spending behavior"

                st.text(f"""
Customer {customer_id}:
Behavior: {behavior}
Reason: {reason}
----------------------------------------
""")

        except Exception as e:
            st.error(f"❌ Error: {e}")
