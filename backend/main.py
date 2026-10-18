"""
FastAPI Backend for Telco Customer Churn Prediction
Provides:
- POST /predict: Predict churn probability, risk level, and top SHAP reasons
- GET /model-info: Model metrics, feature importances, ROC curve, and comparisons
- GET /customer-examples: Pre-configured sample customer profiles for quick UI demos
- Static frontend mounting for full self-hosted single-command execution
- CORS enabled for cross-origin frontend requests
"""

import os
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import shap

# Multi-environment path resolution (works locally, in Docker, and on Vercel Serverless)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(*relative_candidates) -> str:
    search_roots = [
        BASE_DIR,
        THIS_DIR,
        os.getcwd(),
        '/var/task',
        '/var/task/backend',
        '/var/task/data'
    ]
    for rel in relative_candidates:
        for root in search_roots:
            candidate = os.path.normpath(os.path.join(root, rel))
            if os.path.exists(candidate):
                return candidate
    return os.path.normpath(os.path.join(BASE_DIR, relative_candidates[0]))

MODEL_PATH = resolve_path('backend/model.joblib', 'model.joblib')
ARTIFACTS_PATH = resolve_path('backend/model_artifacts.joblib', 'model_artifacts.joblib')
MODEL_INFO_PATH = resolve_path('data/model_info.json', 'model_info.json')
EXAMPLES_PATH = resolve_path('data/customer_examples.json', 'customer_examples.json')
DATA_DIR = resolve_path('data')
FRONTEND_DIR = resolve_path('frontend')

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Machine learning API for predicting telecommunications customer churn with SHAP explanations.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
pipeline = None
explainer = None
artifacts = None
model_info = {}
customer_examples = []
clean_feature_names = []


def load_model_assets():
    global pipeline, explainer, artifacts, model_info, customer_examples, clean_feature_names
    try:
        if os.path.exists(MODEL_PATH):
            pipeline = joblib.load(MODEL_PATH)
            # Create SHAP TreeExplainer for the XGBoost classifier in the pipeline
            xgb_clf = pipeline.named_steps['clf']
            explainer = shap.TreeExplainer(xgb_clf)
            print("Loaded model pipeline and initialized SHAP TreeExplainer.")
            
        if os.path.exists(ARTIFACTS_PATH):
            artifacts = joblib.load(ARTIFACTS_PATH)
            clean_feature_names = artifacts.get('clean_feature_names', [])
            print(f"Loaded artifacts with {len(clean_feature_names)} feature names.")

        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH, 'r') as f:
                model_info = json.load(f)
            print("Loaded model_info.json.")

        if os.path.exists(EXAMPLES_PATH):
            with open(EXAMPLES_PATH, 'r') as f:
                customer_examples = json.load(f)
            print("Loaded customer_examples.json.")
    except Exception as e:
        print(f"Error loading model assets: {e}")


# Load on startup
load_model_assets()


# Pydantic Schemas
class CustomerInput(BaseModel):
    gender: str = Field(default="Female", description="Gender: Female, Male")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="Senior Citizen: 0 or 1")
    Partner: str = Field(default="No", description="Partner: Yes, No")
    Dependents: str = Field(default="No", description="Dependents: Yes, No")
    tenure: int = Field(default=12, ge=0, le=100, description="Months as customer")
    PhoneService: str = Field(default="Yes", description="Phone Service: Yes, No")
    MultipleLines: str = Field(default="No", description="Multiple Lines: Yes, No, No phone service")
    InternetService: str = Field(default="Fiber optic", description="Internet: DSL, Fiber optic, No")
    OnlineSecurity: str = Field(default="No", description="Online Security: Yes, No, No internet service")
    OnlineBackup: str = Field(default="No", description="Online Backup: Yes, No, No internet service")
    DeviceProtection: str = Field(default="No", description="Device Protection: Yes, No, No internet service")
    TechSupport: str = Field(default="No", description="Tech Support: Yes, No, No internet service")
    StreamingTV: str = Field(default="Yes", description="Streaming TV: Yes, No, No internet service")
    StreamingMovies: str = Field(default="Yes", description="Streaming Movies: Yes, No, No internet service")
    Contract: str = Field(default="Month-to-month", description="Contract: Month-to-month, One year, Two year")
    PaperlessBilling: str = Field(default="Yes", description="Paperless Billing: Yes, No")
    PaymentMethod: str = Field(
        default="Electronic check",
        description="Payment: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)"
    )
    MonthlyCharges: float = Field(default=85.0, ge=0, description="Monthly bill in USD")
    TotalCharges: Optional[float] = Field(default=None, ge=0, description="Total cumulative bill in USD")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.5,
                "TotalCharges": 268.5
            }
        }


class ChurnReason(BaseModel):
    feature: str
    impact: float
    direction: str  # "increase" or "decrease"
    human_explanation: str


class PredictResponse(BaseModel):
    probability: float
    risk_level: str  # "Low", "Medium", "High"
    optimal_threshold: float
    predicted_churn: bool
    top_reasons: List[ChurnReason]


def generate_human_explanation(clean_feat: str, val: float, cust_dict: Dict[str, Any]) -> str:
    """Generate clear, context-aware business explanation for SHAP feature attribution."""
    direction_word = "increases" if val > 0 else "reduces"
    contract_val = cust_dict.get('Contract', '')
    internet_val = cust_dict.get('InternetService', '')
    payment_val = cust_dict.get('PaymentMethod', '')
    tech_val = cust_dict.get('TechSupport', '')
    security_val = cust_dict.get('OnlineSecurity', '')
    tenure_val = cust_dict.get('tenure', 0)
    monthly_val = cust_dict.get('MonthlyCharges', 0.0)

    if "Contract_Month-to-month" in clean_feat:
        if contract_val == 'Month-to-month':
            return "Month-to-month contract strongly increases churn risk (no long-term commitment)."
        else:
            return "Having a 1- or 2-year contract significantly protects against churn."
    elif "Contract_Two year" in clean_feat:
        if contract_val == 'Two year':
            return "Two-year contract strongly locks in customer loyalty and reduces churn."
        else:
            return f"{clean_feat.replace('_', ': ')} {direction_word} churn risk."
    elif "Contract_One year" in clean_feat:
        return f"One-year contract {direction_word} churn risk."
    elif clean_feat == "tenure":
        if tenure_val <= 12:
            return f"Short tenure ({tenure_val} months) increases churn risk during onboarding."
        else:
            return f"Established loyalty ({tenure_val} months tenure) substantially reduces churn risk."
    elif "InternetService_Fiber optic" in clean_feat:
        if internet_val == 'Fiber optic':
            return "Fiber optic plan increases churn risk due to high price sensitivity and competitor offers."
        else:
            return "Not having fiber optic reduces churn propensity."
    elif "InternetService_No" in clean_feat:
        return f"Phone-only service (no internet) {direction_word} churn risk."
    elif clean_feat == "MonthlyCharges":
        if monthly_val >= 75.0:
            return f"Higher monthly bill (${monthly_val:.2f}) elevates price sensitivity and churn risk."
        else:
            return f"Affordable monthly bill (${monthly_val:.2f}) helps retain the customer."
    elif clean_feat == "TotalCharges":
        return f"Cumulative spend (${cust_dict.get('TotalCharges', 0):.2f}) {direction_word} churn probability."
    elif "PaymentMethod_Electronic check" in clean_feat:
        if payment_val == 'Electronic check':
            return "Electronic check payment exhibits the highest churn rate across billing methods."
        else:
            return "Automated payment method reduces churn risk compared to manual checks."
    elif "TechSupport_No" in clean_feat:
        if tech_val == 'No':
            return "Lack of tech support increases frustration and churn risk."
        else:
            return "Having active tech support helps retain the customer."
    elif "TechSupport_Yes" in clean_feat:
        return f"Active technical support {direction_word} customer churn risk."
    elif "OnlineSecurity_No" in clean_feat:
        if security_val == 'No':
            return "No online security add-on increases customer churn likelihood."
        else:
            return "Active security add-on increases service stickiness and lowers churn."
    elif "OnlineSecurity_Yes" in clean_feat:
        return f"Active online security {direction_word} churn risk."
    elif "PaperlessBilling_Yes" in clean_feat:
        return f"Paperless billing {direction_word} churn risk."
    elif "SeniorCitizen" in clean_feat:
        return f"Senior citizen segment {direction_word} churn risk."
    else:
        nice_feat = clean_feat.replace('_', ': ')
        return f"{nice_feat} {direction_word} churn probability."


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "explainer_loaded": explainer is not None,
        "features_count": len(clean_feature_names)
    }


@app.get("/model-info")
@app.get("/api/model-info")
def get_model_info():
    """Return model metrics, comparison table, ROC curve, and top features for frontend."""
    if not model_info:
        # Fallback reload
        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH, 'r') as f:
                return json.load(f)
        raise HTTPException(status_code=404, detail="Model info not yet generated.")
    return model_info


@app.get("/customer-examples")
@app.get("/api/customer-examples")
def get_customer_examples():
    """Return pre-configured sample customer profiles for interactive demos."""
    if not customer_examples:
        if os.path.exists(EXAMPLES_PATH):
            with open(EXAMPLES_PATH, 'r') as f:
                return json.load(f)
        raise HTTPException(status_code=404, detail="Customer examples not found.")
    return customer_examples


@app.post("/predict", response_model=PredictResponse)
@app.post("/api/predict", response_model=PredictResponse)
def predict_churn(customer: CustomerInput):
    """
    Accepts customer profile JSON.
    Returns churn probability, risk level, optimal threshold decision, and top SHAP drivers.
    """
    if pipeline is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model is still initializing or not found.")

    # Convert Pydantic model to dict
    cust_dict = customer.dict()

    # Handle TotalCharges if not provided or empty
    if cust_dict['TotalCharges'] is None or cust_dict['TotalCharges'] <= 0:
        cust_dict['TotalCharges'] = float(cust_dict['tenure'] * cust_dict['MonthlyCharges'])

    df_input = pd.DataFrame([cust_dict])

    try:
        # 1. Predict Probability
        prob = float(pipeline.predict_proba(df_input)[0, 1])

        # 2. Risk Level & Decision
        opt_th = artifacts.get('optimal_threshold', 0.25) if artifacts else 0.25
        
        if prob >= 0.60:
            risk_level = "High"
        elif prob >= 0.30:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        predicted_churn = bool(prob >= opt_th)

        # 3. Compute SHAP values
        preprocessor = pipeline.named_steps['prep']
        transformed_input = preprocessor.transform(df_input)
        shap_vals = explainer(transformed_input).values[0]

        # Extract top 4 reasons by absolute SHAP attribution
        top_indices = np.argsort(np.abs(shap_vals))[::-1][:4]
        
        top_reasons = []
        for idx in top_indices:
            feat_name = clean_feature_names[idx] if idx < len(clean_feature_names) else f"feature_{idx}"
            val = float(shap_vals[idx])
            direction = "increase" if val > 0 else "decrease"
            
            human_text = generate_human_explanation(feat_name, val, cust_dict)

            clean_label = feat_name.replace('_', ': ')
            top_reasons.append(ChurnReason(
                feature=clean_label,
                impact=round(val, 3),
                direction=direction,
                human_explanation=human_text
            ))

        return PredictResponse(
            probability=round(prob, 4),
            risk_level=risk_level,
            optimal_threshold=round(float(opt_th), 2),
            predicted_churn=predicted_churn,
            top_reasons=top_reasons
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Serve static data and frontend files if directories exist
if os.path.exists(DATA_DIR):
    app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

if os.path.exists(FRONTEND_DIR):
    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
