import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(layout="wide")
st.title("📊 Customer Behavior Prediction System")

# ------------------ LOAD DATA ------------------
st.header("📁 Upload Data")

customers_file = st.file_uploader("Upload Customers Dataset", type=["csv"])
orders_file = st.file_uploader("Upload Orders Dataset", type=["csv"])

if customers_file and orders_file:

    customers = pd.read_csv(customers_file)
    orders = pd.read_csv(orders_file)

    st.success("Datasets Loaded ✅")

    st.write("Customers Data", customers.head())
    st.write("Orders Data", orders.head())

    # ------------------ MERGE ------------------
    st.header("🔗 Merge Data")

    if "Customer ID" not in customers.columns or "Customer ID" not in orders.columns:
        st.error("❌ 'Customer ID' column required in both datasets")
        st.stop()

    df = pd.merge(customers, orders, on="Customer ID")

    st.write("Merged Dataset", df.head())

    # ------------------ FEATURE CREATION ------------------
    st.header("⚙️ Feature Engineering")

    # Example features (based on your project)
    if "Purchase Amount" in df.columns:
        df["Spending_Level"] = pd.cut(
            df["Purchase Amount"],
            bins=[0, 500, 2000, 10000],
            labels=["Low", "Medium", "High"]
        )

    st.write(df.head())

    # ------------------ TARGET ------------------
    target_col = st.selectbox("Select Target Column (Behavior)", df.columns)

    # ------------------ MODEL SELECTION ------------------
    st.header("🤖 Select Models")

    selected_models = st.multiselect(
        "Choose Models",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        default=["Random Forest"]
    )

    # ------------------ TRAIN ------------------
    if st.button("🚀 Train Models"):

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
            X_test_scaled = scaler.transform(X_test)

            # Models
            model_dict = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }

            results = []
            best_model = None
            best_score = 0
            best_pred = None
            best_name = ""

            for name in selected_models:

                model = model_dict[name]
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test_scaled)

                train_acc = model.score(X_train, y_train)
                test_acc = accuracy_score(y_test, y_pred)

                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                results.append([name, train_acc, test_acc, prec, rec, f1])

                if test_acc > best_score:
                    best_score = test_acc
                    best_model = model
                    best_pred = y_pred
                    best_name = name

            # ------------------ SHOW RESULTS ------------------
            results_df = pd.DataFrame(results, columns=[
                "Model", "Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1 Score"
            ])

            st.subheader("📊 Model Evaluation")
            st.dataframe(results_df)

            st.success(f"🏆 Best Model: {best_name}")

            # ------------------ OUTPUT ------------------
            st.subheader("📦 Customer Behavior Output")

            for i in range(len(best_pred)):

                customer_id = id_test.iloc[i]
                behavior = labels[best_pred[i]]

                # Smart reasoning
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
