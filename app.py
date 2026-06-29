import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Model load
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Customer Churn Prediction")
st.markdown("Predict whether a bank customer will **leave or stay**")
st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    is_active = st.selectbox("Is Active Member", ["Yes", "No"])

st.divider()

# Encode inputs
gender_val = 1 if gender == "Male" else 0
is_active_val = 1 if is_active == "Yes" else 0
geo_spain = 1 if geography == "Spain" else 0
geo_germany = 1 if geography == "Germany" else 0

# Predict button
if st.button("🔍 Predict", use_container_width=True):
    
    input_data = pd.DataFrame({
        'Age': [age],
        'IsActiveMember': [is_active_val],
        'NumOfProducts': [num_products],
        'Balance': [balance],
        'EstimatedSalary': [estimated_salary],
        'CreditScore': [credit_score],
        'Tenure': [tenure],
        'Geography_Spain': [geo_spain],
        'Geography_Germany': [geo_germany],
        'Gender': [gender_val]
    })
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    churn_prob = probability[0][1] * 100
    stay_prob = probability[0][0] * 100
    
    st.subheader("📊 Prediction Result")
    
    if prediction[0] == 1:
        st.error(f"⚠️ This customer is likely to CHURN!")
    else:
        st.success(f"✅ This customer will likely STAY!")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Churn Probability", f"{churn_prob:.2f}%")
    with col4:
        st.metric("Stay Probability", f"{stay_prob:.2f}%")
    
    st.progress(int(churn_prob))