import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="centered"
)

# Load model and scaler
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# Title
st.title("🏦 Customer Churn Predictor")
st.markdown("Predict whether a bank customer will **exit or stay** using ML.")
st.divider()

# Input form
st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)
    balance = st.number_input("Account Balance (₹)", min_value=0.0, value=50000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])

with col2:
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary (₹)", min_value=0.0, value=50000.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

st.divider()

# Predict button
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

    # Encode inputs
    gender_encoded = 1 if gender == "Female" else 0
    has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
    is_active_encoded = 1 if is_active == "Yes" else 0
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    # Feature array (same order as training)
    features = np.array([[
        credit_score, gender_encoded, age, tenure, balance,
        num_products, has_cr_card_encoded, is_active_encoded,
        estimated_salary, geo_germany, geo_spain
    ]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to CHURN!**")
        st.metric("Churn Probability", f"{probability[1]*100:.1f}%")
        st.markdown("""
        **Suggested Actions:**
        - 📞 Reach out with a retention offer
        - 🎁 Offer loyalty rewards or cashback
        - 💬 Schedule a customer satisfaction call
        """)
    else:
        st.success(f"✅ This customer is **likely to STAY!**")
        st.metric("Retention Probability", f"{probability[0]*100:.1f}%")
        st.markdown("""
        **Status:** Customer is satisfied. Keep up the good service! 😊
        """)

    # Probability bar
    st.subheader("📈 Probability Breakdown")
    col_a, col_b = st.columns(2)
    col_a.metric("Stay (0)", f"{probability[0]*100:.1f}%")
    col_b.metric("Churn (1)", f"{probability[1]*100:.1f}%")
    st.progress(float(probability[1]))

# Footer
st.divider()
st.caption("Built with ❤️ using Logistic Regression + SMOTE | Final Year Project")
