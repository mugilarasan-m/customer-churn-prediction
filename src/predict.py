import os
import joblib
import pandas as pd
from typing import Dict, Any, Tuple

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'customer_churn_model.pkl')

def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run src/train_model.py first.")
    return joblib.load(model_path)

def predict_single_customer(customer_data: Dict[str, Any], model=None) -> Dict[str, Any]:
    if model is None:
        model = load_model()

    df_input = pd.DataFrame([customer_data])
    churn_proba = float(model.predict_proba(df_input)[0, 1])

    prediction = "Likely to Churn" if churn_proba >= 0.5 else "Not Likely to Churn"

    if churn_proba < 0.30:
        risk_level = "Low Risk"
    elif churn_proba < 0.65:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return {
        "churn_prediction": prediction,
        "churn_probability": round(churn_proba * 100, 2),
        "risk_level": risk_level
    }

if __name__ == "__main__":
    sample_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.7,
        'TotalCharges': 151.65
    }

    result = predict_single_customer(sample_customer)
    print("Inference Test Result:", result)