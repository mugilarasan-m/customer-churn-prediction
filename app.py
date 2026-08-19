import streamlit as st
import pandas as pd
from src.predict import predict_single_customer, load_model

st.set_page_config(
    page_title="Customer Churn Prediction System",
    layout="wide"
)

st.title("Customer Churn Intelligence System")
st.markdown("Predict customer churn probabilities, assess risk levels, and receive data-driven retention strategies.")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}. Please train the model using `python src/train_model.py` first.")
    model_loaded = False

if model_loaded:
    st.subheader("1. Customer Profile Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    with col3:
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.markdown("---")
    st.subheader("2. Financial Information")

    f_col1, f_col2 = st.columns(2)

    with f_col1:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=300.0,
            value=65.0,
            step=0.5
        )

    with f_col2:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=780.0,
            step=1.0
        )

    if st.button("Predict Churn Risk", type="primary", use_container_width=True):
        payload = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        result = predict_single_customer(payload, model=model)

        st.markdown("---")
        st.subheader("3. Prediction & Risk Assessment")

        res_col1, res_col2, res_col3 = st.columns(3)

        res_col1.metric("Prediction Status", result["churn_prediction"])
        res_col2.metric("Churn Probability", f"{result['churn_probability']}%")
        res_col3.metric("Risk Classification", result["risk_level"])

        if result["risk_level"] == "High Risk":
            st.error("High Risk Alert! Action Recommended:")
            st.write("- **Incentive Offer:** Offer a 1-year or 2-year contract discount to mitigate month-to-month churn.")
            st.write("- **Value Add:** Bundle complimentary Tech Support or Online Security.")
            st.write("- **Outreach:** Schedule a customer success check-in call.")

        elif result["risk_level"] == "Medium Risk":
            st.warning("Medium Risk: Monitor Customer Engagement:")
            st.write("- Provide tailored product updates and verify customer satisfaction with internet speeds.")
            st.write("- Offer digital payment discounts to transition off paper check methods.")

        else:
            st.success("Low Risk: Customer is Stable:")
            st.write("- Eligible for upselling premium add-ons or loyalty rewards.")