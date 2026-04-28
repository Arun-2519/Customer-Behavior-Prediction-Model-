import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

st.set_page_config(layout="wide")
st.title("📊 Customer Behavior Classification (2 Dataset Merge)")

# ------------------ STEP 1: UPLOAD DATA ------------------
st.header("Step 1: Upload Datasets")

file1 = st.file_uploader("Upload Dataset 1", type=["csv"])
file2 = st.file_uploader("Upload Dataset 2", type=["csv"])

if file1 and file2:
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    st.success("Datasets Loaded ✅")
    st.write("Dataset 1", df1.head())
    st.write("Dataset 2", df2.head())

    # ------------------ STEP 2: MERGE ------------------
    st.header("Step 2: Merge Datasets")

    common_cols = list(set(df1.columns) & set(df2.columns))

    if len(common_cols) == 0:
        st.error("❌ No common column to merge")
        st.stop()

    merge_col = st.selectbox("Select Common Column", common_cols)

    df = pd.merge(df1, df2, on=merge_col)

    st.success("Merged Dataset ✅")
    st.write(df.head())

    # ------------------ STEP 3: SELECT TARGET ------------------
    st.header("Step 3: Select Target Column")

    target_col = st.selectbox("Select Target (Classification Column)", df.columns)

    # ------------------ STEP 4: PREPROCESS ------------------
    if st.button("🚀 Train Model"):

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Convert categorical → numbers
        X = pd.get_dummies(X, drop_first=True)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ------------------ STEP 5: MODELS ------------------
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            ("Random Forest", RandomForestClassifier(n_estimators=100)),
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

        # ------------------ STEP 6: SHOW RESULTS ------------------
        results_df = pd.DataFrame(results, columns=[
            "Model", "Accuracy", "Precision", "Recall", "F1 Score"
        ])

        st.subheader("📊 Model Comparison")
        st.dataframe(results_df)

        st.success(f"🏆 Best Model: {best_name}")

        # ------------------ STEP 7: SAVE MODEL ------------------
        model_data = {
            "model": best_model,
            "scaler": scaler,
            "columns": X.columns.tolist(),
            "target": target_col
        }

        with open("classification_model.pkl", "wb") as f:
            pickle.dump(model_data, f)

        with open("classification_model.pkl", "rb") as f:
            st.download_button("📥 Download Model", f, "classification_model.pkl")

    # ------------------ STEP 8: PREDICTION ------------------
    if st.button("🔮 Predict Using Saved Model"):
        try:
            with open("classification_model.pkl", "rb") as f:
                data = pickle.load(f)

            model = data["model"]
            scaler = data["scaler"]
            saved_cols = data["columns"]
            target_col = data["target"]

            X = df.drop(columns=[target_col])
            X = pd.get_dummies(X, drop_first=True)

            for col in saved_cols:
                if col not in X:
                    X[col] = 0

            X = X[saved_cols]
            X_scaled = scaler.transform(X)

            preds = model.predict(X_scaled)

            df["Prediction"] = preds

            st.subheader("🔮 Predictions")
            st.write(df.head(20))

        except Exception as e:
            st.error(e)
