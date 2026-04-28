import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide")
st.title("📊 Customer Behavior Prediction")

# ------------------ UPLOAD ------------------
customers_file = st.file_uploader("Upload Customers Dataset", type=["csv"])
orders_file = st.file_uploader("Upload Orders Dataset", type=["csv"])

if customers_file and orders_file:

    customers = pd.read_csv(customers_file)
    orders = pd.read_csv(orders_file)

    st.success("Datasets Loaded ✅")

    st.write("Customers Data", customers.head())
    st.write("Orders Data", orders.head())

    # ------------------ FIX CUSTOMER ID ------------------
    def find_customer_id_column(df):
        for col in df.columns:
            if "customer" in col.lower() and "id" in col.lower():
                return col
        return None

    cust_id_col = find_customer_id_column(customers)

    if cust_id_col is None:
        st.error("❌ Customer ID column not found in customers dataset")
        st.stop()

    # Rename to standard name
    customers.rename(columns={cust_id_col: "Customer ID"}, inplace=True)

    # ------------------ SAME NOTEBOOK LOGIC ------------------
    # Add Customer ID to orders (UNCHANGED LOGIC)
    if len(customers) != len(orders):
        st.warning("⚠️ Row mismatch → trimming to match size")
        min_len = min(len(customers), len(orders))
        customers = customers.head(min_len)
        orders = orders.head(min_len)

    orders["Customer ID"] = customers["Customer ID"]

    # Merge
    df = pd.merge(customers, orders, on="Customer ID")

    st.write("Merged Data", df.head())

    # ------------------ FEATURE ------------------
    if "Purchase Amount" in df.columns:
        df["Spending Level"] = pd.cut(
            df["Purchase Amount"],
            bins=[0, 500, 2000, 10000],
            labels=["Low", "Medium", "High"]
        )

    # ------------------ TARGET ------------------
    target_col = st.selectbox("Select Target Column", df.columns)

    # ------------------ TRAIN ------------------
    if st.button("🚀 Train Model"):

        try:
            df = df.dropna()

            X = df.drop(columns=[target_col])
            y = df[target_col]

            X = pd.get_dummies(X)
            y_encoded, labels = pd.factorize(y)

            X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
                X, y_encoded, df["Customer ID"], test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # ------------------ METRICS ------------------
            st.subheader("📊 Model Performance")
            st.write("Train Accuracy:", model.score(X_train, y_train))
            st.write("Test Accuracy:", accuracy_score(y_test, y_pred))

            # ------------------ OUTPUT ------------------
            st.subheader("📦 Customer Behavior Output")

            for i in range(len(y_pred)):

                customer_id = id_test.iloc[i]
                behavior = labels[y_pred[i]]

                # SAME REASON LOGIC
                if "Purchase Amount" in df.columns:
                    purchase = df.iloc[id_test.index[i]]["Purchase Amount"]

                    if purchase > df["Purchase Amount"].mean():
                        reason = "Very consistent spending"
                    else:
                        reason = "Irregular spending behavior"
                else:
                    reason = "Based on model prediction"

                st.text(f"""
Customer {customer_id}:
Behavior: {behavior}
Reason: {reason}
----------------------------------------
""")

        except Exception as e:
            st.error(f"❌ Error: {e}")
