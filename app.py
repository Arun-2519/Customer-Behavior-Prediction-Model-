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
st.title("📊 Customer Behavior Prediction")

# ------------------ UPLOAD ------------------
file1 = st.file_uploader("Upload Dataset 1", type=["csv"])
file2 = st.file_uploader("Upload Dataset 2", type=["csv"])

if file1 and file2:

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    st.success("Datasets Loaded ✅")

    # ------------------ MERGE ------------------
    common_cols = list(set(df1.columns) & set(df2.columns))

    if not common_cols:
        st.error("❌ No common column found")
        st.stop()

    merge_col = st.selectbox("Select Common Column", common_cols)

    df = pd.merge(df1, df2, on=merge_col)

    st.write("Merged Data", df.head())

    # ------------------ TARGET ------------------
    target_col = st.selectbox("Select Target Column (Behavior)", df.columns)

    # ------------------ MODEL SELECTION ------------------
    st.subheader("Select Models")

    selected_models = st.multiselect(
        "Choose Models",
        ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"],
        default=["Random Forest"]
    )

    # ------------------ TRAIN ------------------
    if st.button("🚀 Train Models"):

        try:
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

            # Model dictionary
            model_dict = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "SVM": SVC()
            }

            results = []
            best_model = None
            best_score = 0
            best_pred = None
            best_name = ""

            # ------------------ TRAIN LOOP ------------------
            for name in selected_models:
                model = model_dict[name]

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test_scaled)

                train_acc = model.score(X_train, y_train)
                test_acc = accuracy_score(y_test, y_pred)

                prec = precision_score(y_test, y_pred, average="weighted")
                rec = recall_score(y_test, y_pred, average="weighted")
                f1 = f1_score(y_test, y_pred, average="weighted")

                results.append([name, train_acc, test_acc, prec, rec, f1])

                if test_acc > best_score:
                    best_score = test_acc
                    best_model = model
                    best_pred = y_pred
                    best_name = name

            # ------------------ RESULTS ------------------
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

                # Safe reason logic
                if "Purchase Amount" in df.columns:
                    purchase = df.iloc[id_test.index[i]]["Purchase Amount"]

                    if purchase > df["Purchase Amount"].mean():
                        reason = "Very consistent spending"
                    else:
                        reason = "Irregular spending behavior"
                else:
                    reason = "Behavior based on model prediction"

                st.text(f"""
Customer {customer_id}:
Behavior: {behavior}
Reason: {reason}
----------------------------------------
""")

        except Exception as e:
            st.error(f"Error: {e}")
