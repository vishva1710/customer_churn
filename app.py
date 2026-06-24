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

# Header
st.title("🏦 Customer Churn Predictor")
st.markdown("Predict whether a bank customer will **stay or leave** based on their profile.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 95, 35)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)

with col2:
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0)

st.divider()

# Predict button
if st.button("🔍 Predict", use_container_width=True, type="primary"):

    # Encode inputs
    geo_map = {"France": 0, "Germany": 1, "Spain": 2}
    gender_map = {"Male": 1, "Female": 0}
    yes_no_map = {"Yes": 1, "No": 0}

    input_data = np.array([[
        credit_score,
        geo_map[geography],
        gender_map[gender],
        age,
        tenure,
        balance,
        num_of_products,
        yes_no_map[has_cr_card],
        yes_no_map[is_active_member],
        estimated_salary
    ]])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **Customer is likely to CHURN**")
        st.metric("Churn Probability", f"{probability[1]*100:.1f}%")
        st.markdown("""
        **Suggested Actions:**
        - Offer loyalty rewards or discounts
        - Assign a relationship manager
        - Review account activity and reach out proactively
        """)
    else:
        st.success(f"✅ **Customer is likely to STAY (Retained)**")
        st.metric("Retention Probability", f"{probability[0]*100:.1f}%")
        st.markdown("""
        **Customer Status:**
        - Low churn risk
        - Continue regular engagement
        - Consider upselling additional products
        """)

    # Probability bar
    st.subheader("Probability Breakdown")
    col_a, col_b = st.columns(2)
    col_a.metric("🟢 Retained", f"{probability[0]*100:.1f}%")
    col_b.metric("🔴 Churned", f"{probability[1]*100:.1f}%")
    st.progress(float(probability[1]))

st.divider()
st.caption("Model: Random Forest Classifier | Accuracy: 81.56% | Dataset: 10,000 bank customers")
