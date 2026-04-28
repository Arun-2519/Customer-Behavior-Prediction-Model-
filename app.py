import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

st.set_page_config(layout="wide")
st.title("📊 Customer Behavior Prediction (Advanced Training)")

# ------------------ STEP 1: UPLOAD ------------------
file1 = st.file_uploader("Upload Dataset 1", type=["csv"])
file2 = st.file_uploader("Upload Dataset 2", type=["csv"])

if file1 and file2:
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    st.success("Datasets Loaded ✅")

    # ------------------ STEP 2: MERGE ------------------
    common_cols = list(set(df1.columns) & set(df2.columns))

    if len(common_cols) == 0:
        st.error("❌ No common column")
        st.stop()

    merge_col = st.selectbox("Select Common Column", common_cols)

    df = pd.merge(df1, df2, on=merge_col)
    st.write(df.head())

    # ------------------ STEP 3: TARGET ------------------
    target_col = st.selectbox("Select Behavior Column", df.columns)

    # ------------------ STEP 4: TRAIN ------------------
    if st.button("🚀 Train Models"):

        X = df.drop(columns=[target_col])
        y = df[target_col]
        customer_ids = df[merge_col]

        # Convert categorical
        X = pd.get_dummies(X, drop_first=True)
        y_encoded, labels = pd.factorize(y)

        # Split
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X, y_encoded, customer_ids, test_size=0.2, random_state=42
        )

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ------------------ MODELS ------------------
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            ("Random Forest", RandomForestClassifier()),
            ("Gradient Boosting", GradientBoostingClassifier()),
            ("SVM", SVC())
        ]

        results = []
        best_model = None
        best_score = 0
        best_name = ""

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            results.append([name, acc, prec, rec, f1])

            if acc > best_score:
                best_score = acc
                best_model = model
                best_name = name
                best_pred = y_pred

        # ------------------ SHOW RESULTS ------------------
        results_df = pd.DataFrame(results, columns=[
            "Model", "Accuracy", "Precision", "Recall", "F1 Score"
        ])

        st.subheader("📊 Model Comparison")
        st.dataframe(results_df)

        st.success(f"🏆 Best Model: {best_name}")

        # ------------------ OUTPUT ------------------
        st.subheader("📦 Customer Behavior Output")

        for i in range(len(best_pred)):

            customer_id = id_test.iloc[i]
            behavior = labels[best_pred[i]]

            # Reason logic
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
