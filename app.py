import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("Customer Behavior Prediction (2 Dataset Merge)")

# ------------------ STEP 1: UPLOAD ------------------
st.header(" Upload Datasets")

file1 = st.file_uploader("Upload Dataset 1", type=["csv"])
file2 = st.file_uploader("Upload Dataset 2", type=["csv"])

if file1 and file2:
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    st.success("Datasets Loaded ")

    st.write("Dataset 1", df1.head())
    st.write("Dataset 2", df2.head())

    # ------------------ STEP 2: MERGE ------------------
    st.header(" Merge Datasets")

    common_cols = list(set(df1.columns) & set(df2.columns))

    if len(common_cols) == 0:
        st.error(" No common column found")
        st.stop()

    merge_col = st.selectbox("Select Common Column", common_cols)

    df = pd.merge(df1, df2, on=merge_col)

    st.success("Merged Successfully ")
    st.write(df.head())

    # ------------------ STEP 3: SELECT TARGET ------------------
    st.header("Select Target Column")

    target_col = st.selectbox("Select Behavior Column", df.columns)

    # ------------------ STEP 4: TRAIN MODEL ------------------
    if st.button("Train Model"):

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
        X_test = scaler.transform(X_test)

        # Model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        st.success("Model Trained ")

        # ------------------ STEP 5: OUTPUT ------------------
        st.subheader(" Customer Behavior Output")

        for i in range(len(y_pred)):

            customer_id = id_test.iloc[i]
            behavior = labels[y_pred[i]]

            # 🔥 REASON LOGIC
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

    # ------------------ STEP 6: PREDICT FULL DATA ------------------
    if st.button(" Predict Full Dataset"):

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X = pd.get_dummies(X, drop_first=True)
        y_encoded, labels = pd.factorize(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier()
        model.fit(X_scaled, y_encoded)

        preds = model.predict(X_scaled)

        df["Predicted Behavior"] = [labels[p] for p in preds]

        st.subheader(" Full Prediction")
        st.dataframe(df.head(20))
