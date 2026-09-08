import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load trained model and files
# -----------------------------

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information below to predict whether "
    "the customer is likely to leave the bank."
)


# -----------------------------
# Input fields
# -----------------------------

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_card = st.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

active_member = st.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)


# -----------------------------
# Prediction button
# -----------------------------

if st.button("Predict Churn"):

    # Convert inputs to model format

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [1 if has_card == "Yes" else 0],
        "IsActiveMember": [1 if active_member == "Yes" else 0],
        "EstimatedSalary": [estimated_salary],

        "Geography_Germany": [
            1 if geography == "Germany" else 0
        ],

        "Geography_Spain": [
            1 if geography == "Spain" else 0
        ],

        "Gender_Male": [
            1 if gender == "Male" else 0
        ]
    })


    # Make sure columns are in exactly the same order
    input_data = input_data.reindex(
        columns=features,
        fill_value=0
    )


    # Scale input
    input_scaled = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]


    # -----------------------------
    # Display result
    # -----------------------------

    if prediction == 1:

        st.error("⚠️ Customer is likely to CHURN")

        st.write(
            f"Churn Probability: **{probability * 100:.2f}%**"
        )

    else:

        st.success("✅ Customer is likely to STAY")

        st.write(
            f"Churn Probability: **{probability * 100:.2f}%**"
        )
        