"""
FastAPI Backend for Telco Customer Churn Prediction & Explainability
Optimized for Vercel Serverless Function & Local Execution.
Zero heavy dependencies (no scipy, nvidia, numba, or pandas required at runtime).
Evaluates the 140-tree Tuned XGBoost model and calculates tree-path feature attribution in pure Python.
"""

import os
import sys
import json
import math
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

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

XGB_MODEL_PATH = resolve_path('backend/xgb_model.json', 'xgb_model.json')
MODEL_INFO_PATH = resolve_path('data/model_info.json', 'model_info.json')
EXAMPLES_PATH = resolve_path('data/customer_examples.json', 'customer_examples.json')
DATA_DIR = resolve_path('data')
FRONTEND_DIR = resolve_path('frontend')

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Machine learning API for predicting telecommunications customer churn with XGBoost & Explainability.",
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

# 46 Feature names exactly matching trained preprocessor
CLEAN_FEATURE_NAMES = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Female', 'gender_Male',
    'SeniorCitizen_0', 'SeniorCitizen_1',
    'Partner_No', 'Partner_Yes',
    'Dependents_No', 'Dependents_Yes',
    'PhoneService_No', 'PhoneService_Yes',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]
FEATURE_TO_IDX = {name: i for i, name in enumerate(CLEAN_FEATURE_NAMES)}

# Model state
xgb_trees = []
xgb_base_margin = -1.018399  # default logit for 26.54% churn rate
model_info = {}
customer_examples = []
scaler_params = {
    'tenure': {'mean': 32.37, 'scale': 24.56},
    'MonthlyCharges': {'mean': 64.76, 'scale': 30.09},
    'TotalCharges': {'mean': 2283.30, 'scale': 2266.77}
}
optimal_threshold = 0.15


def load_model_assets():
    global xgb_trees, xgb_base_margin, model_info, customer_examples, scaler_params, optimal_threshold
    try:
        if os.path.exists(XGB_MODEL_PATH):
            with open(XGB_MODEL_PATH, 'r') as f:
                xgb_data = json.load(f)
            xgb_trees = xgb_data['learner']['gradient_booster']['model']['trees']
            base_score_str = xgb_data['learner']['learner_model_param'].get('base_score', '[0.2653532]')
            base_score = float(base_score_str.strip('[]'))
            xgb_base_margin = math.log(base_score / (1.0 - base_score))
            print(f"Loaded {len(xgb_trees)} XGBoost trees with base margin {xgb_base_margin:.4f}.")

        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH, 'r') as f:
                model_info = json.load(f)
            if 'best_model' in model_info:
                optimal_threshold = float(model_info['best_model'].get('optimal_threshold', 0.15))
            if 'client_scoring_fallback' in model_info and 'scaler' in model_info['client_scoring_fallback']:
                scaler_params = model_info['client_scoring_fallback']['scaler']
            print("Loaded model_info.json and scaler parameters.")

        if os.path.exists(EXAMPLES_PATH):
            with open(EXAMPLES_PATH, 'r') as f:
                customer_examples = json.load(f)
            print(f"Loaded {len(customer_examples)} customer examples.")
    except Exception as e:
        print(f"Warning loading assets: {e}")


# Initialize model on module load
load_model_assets()


# Pydantic Schemas
class CustomerInput(BaseModel):
    gender: str = Field(default="Female")
    SeniorCitizen: int = Field(default=0, ge=0, le=1)
    Partner: str = Field(default="No")
    Dependents: str = Field(default="No")
    tenure: int = Field(default=12, ge=0, le=100)
    PhoneService: str = Field(default="Yes")
    MultipleLines: str = Field(default="No")
    InternetService: str = Field(default="Fiber optic")
    OnlineSecurity: str = Field(default="No")
    OnlineBackup: str = Field(default="No")
    DeviceProtection: str = Field(default="No")
    TechSupport: str = Field(default="No")
    StreamingTV: str = Field(default="Yes")
    StreamingMovies: str = Field(default="Yes")
    Contract: str = Field(default="Month-to-month")
    PaperlessBilling: str = Field(default="Yes")
    PaymentMethod: str = Field(default="Electronic check")
    MonthlyCharges: float = Field(default=85.0, ge=0)
    TotalCharges: Optional[float] = Field(default=None, ge=0)


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


def preprocess_customer_to_vector(cust: Dict[str, Any]) -> List[float]:
    """Encodes raw customer JSON into the 46 features expected by XGBoost."""
    tenure = float(cust.get('tenure', 0))
    monthly = float(cust.get('MonthlyCharges', 0.0))
    raw_total = cust.get('TotalCharges')
    total = float(raw_total) if raw_total is not None and raw_total > 0 else (tenure * monthly)

    vec = [0.0] * 46
    # Scaled numeric features
    vec[0] = (tenure - scaler_params['tenure']['mean']) / scaler_params['tenure']['scale']
    vec[1] = (monthly - scaler_params['MonthlyCharges']['mean']) / scaler_params['MonthlyCharges']['scale']
    vec[2] = (total - scaler_params['TotalCharges']['mean']) / scaler_params['TotalCharges']['scale']

    # One-hot categorical features
    for col, val in cust.items():
        if col not in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            k = f"{col}_{val}"
            if k in FEATURE_TO_IDX:
                vec[FEATURE_TO_IDX[k]] = 1.0

    return vec


def evaluate_xgboost_pure(feature_vector: List[float]):
    """Evaluates the 140 XGBoost decision trees and computes path attributions."""
    raw_score = xgb_base_margin
    attributions = [0.0] * 46

    for tree in xgb_trees:
        left = tree['left_children']
        right = tree['right_children']
        split_idx = tree['split_indices']
        cond = tree['split_conditions']
        node = 0
        path = []
        while left[node] != -1:
            feat = split_idx[node]
            val = feature_vector[feat]
            path.append(feat)
            if val < cond[node]:
                node = left[node]
            else:
                node = right[node]
        leaf_val = cond[node]
        raw_score += leaf_val

        # Distribute tree margin along the path for explainability
        if path:
            weight = leaf_val / len(path)
            for f in path:
                attributions[f] += weight

    # Sigmoid function
    prob = 1.0 / (1.0 + math.exp(-raw_score))
    return prob, attributions


def generate_human_explanation(clean_feat: str, val: float, cust_dict: Dict[str, Any]) -> str:
    """Generate clear, context-aware business explanation for feature attribution."""
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


# Endpoints
@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "trees_count": len(xgb_trees),
        "model_loaded": len(xgb_trees) > 0,
        "features_count": len(CLEAN_FEATURE_NAMES)
    }


@app.get("/model-info")
@app.get("/api/model-info")
def get_model_info():
    """Return model metrics, comparison table, ROC curve, and top features."""
    if not model_info:
        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH, 'r') as f:
                return json.load(f)
        raise HTTPException(status_code=404, detail="Model info not found.")
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
    Returns churn probability, risk level, cost-optimized threshold decision, and top reasons.
    """
    if not xgb_trees:
        load_model_assets()
        if not xgb_trees:
            raise HTTPException(status_code=503, detail="Model assets not loaded.")

    cust_dict = customer.dict()
    if cust_dict['TotalCharges'] is None or cust_dict['TotalCharges'] <= 0:
        cust_dict['TotalCharges'] = float(cust_dict['tenure'] * cust_dict['MonthlyCharges'])

    try:
        vec = preprocess_customer_to_vector(cust_dict)
        prob, attributions = evaluate_xgboost_pure(vec)

        # Risk classification
        if prob >= 0.60:
            risk_level = "High"
        elif prob >= 0.30:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        predicted_churn = bool(prob >= optimal_threshold)

        # Extract top 4 driver features by absolute impact
        top_indices = sorted(range(len(attributions)), key=lambda i: abs(attributions[i]), reverse=True)[:4]
        
        top_reasons = []
        for idx in top_indices:
            feat_name = CLEAN_FEATURE_NAMES[idx]
            impact_val = float(attributions[idx])
            direction = "increase" if impact_val > 0 else "decrease"
            human_text = generate_human_explanation(feat_name, impact_val, cust_dict)
            clean_label = feat_name.replace('_', ': ')

            top_reasons.append(ChurnReason(
                feature=clean_label,
                impact=round(impact_val, 3),
                direction=direction,
                human_explanation=human_text
            ))

        return PredictResponse(
            probability=round(prob, 4),
            risk_level=risk_level,
            optimal_threshold=round(float(optimal_threshold), 2),
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
