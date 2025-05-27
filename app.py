# %%
import streamlit as st
import joblib
import pandas as pd

# %%
model = joblib.load("rf_churn_model.joblib")

# %%


st.title("Customer Churn Prediction")

# Input fields
st.header("Enter Customer Information")
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# Add more features based on your model input

# Make prediction
if st.button("Predict"):
    input_df = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
        # Include all features your model was trained on
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction:")
    st.write("Churn" if prediction == 1 else "Not Churn")
    st.write(f"Probability of Churn: {probability:.2f}")


# %%



