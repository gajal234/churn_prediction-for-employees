import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load saved model and preprocessing objects
saved_objects = pickle.load(open('churn_prediction.pkl', 'rb'))
model = saved_objects["model"]
scaler = saved_objects["scaler"]
label_encoders = saved_objects["label_encoders"]
X_columns = saved_objects["X_columns"]

st.title("Customer Churn Prediction")

# --- INPUT SECTION ---
gender = st.selectbox("Gender", label_encoders["gender"].classes_)
SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
Partner = st.selectbox("Partner", label_encoders["Partner"].classes_)
Dependents = st.selectbox("Dependents", label_encoders["Dependents"].classes_)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
PhoneService = st.selectbox("Phone Service", label_encoders["PhoneService"].classes_)
MultipleLines = st.selectbox("Multiple Lines", label_encoders["MultipleLines"].classes_)
Contract = st.selectbox("Contract", label_encoders["Contract"].classes_)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=0.0)

if st.button("Predict"):
    # Create DataFrame for the input
    data = {
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "Contract": [Contract],
        "TotalCharges": [TotalCharges]
    }
    df1 = pd.DataFrame(data)

    # Encode categorical columns
    for col in ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "Contract"]:
        df1[col] = label_encoders[col].transform(df1[col])

    # Convert SeniorCitizen to numeric
    df1["SeniorCitizen"] = df1["SeniorCitizen"].map({"Yes": 1, "No": 0}) if df1["SeniorCitizen"].dtype == object else df1["SeniorCitizen"]

    # Ensure column order
    df1 = df1[X_columns]

    # Scale features
    df1_scaled = scaler.transform(df1)

    # Predict
    prob = model.predict_proba(df1_scaled)[0][1]  # Probability of churn
    result = model.predict(df1_scaled)[0]

    if result == 0:
        st.success(f"Customer is NOT likely to churn (probability: {prob:.2f})") 
    else:
        st.error(f"Customer is likely to churn (probability: {prob:.2f})") 

    # --- VISUALIZATION SECTION ---
    st.subheader("Logistic vs Linear Regression Curve (Tenure vs Churn Probability)")

    # Create a range of tenure values
    tenure_values = np.linspace(0, 100, 200)
    df_plot = pd.DataFrame({
        "gender": [label_encoders["gender"].transform([gender])[0]] * len(tenure_values),
        "SeniorCitizen": [1 if SeniorCitizen == "Yes" else 0] * len(tenure_values),
        "Partner": [label_encoders["Partner"].transform([Partner])[0]] * len(tenure_values),
        "Dependents": [label_encoders["Dependents"].transform([Dependents])[0]] * len(tenure_values),
        "tenure": tenure_values,
        "PhoneService": [label_encoders["PhoneService"].transform([PhoneService])[0]] * len(tenure_values),
        "MultipleLines": [label_encoders["MultipleLines"].transform([MultipleLines])[0]] * len(tenure_values),
        "Contract": [label_encoders["Contract"].transform([Contract])[0]] * len(tenure_values),
        "TotalCharges": [TotalCharges] * len(tenure_values)
    })

    # Reorder and scale
    df_plot = df_plot[X_columns]
    df_plot_scaled = scaler.transform(df_plot)

    # Logistic regression predictions
    logistic_probs = model.predict_proba(df_plot_scaled)[:, 1]

    # Linear regression just for comparison
    lin_reg = LinearRegression()
    lin_reg.fit(df_plot[["tenure"]], logistic_probs)
    linear_preds = lin_reg.predict(df_plot[["tenure"]])

    # Plot
    fig, ax = plt.subplots()
    ax.plot(tenure_values, logistic_probs, color="blue", label="Logistic Regression")
    ax.plot(tenure_values, linear_preds, color="red", linestyle="--", label="Linear Regression")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn Probability")
    ax.set_title("Logistic vs Linear Regression")
    ax.legend()

    st.pyplot(fig)
