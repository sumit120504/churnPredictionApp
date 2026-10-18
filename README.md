# 🛡️ ChurnShield AI: Customer Churn Prediction & Explainability Web App

An end-to-end Machine Learning web application designed to predict telecommunications customer churn, explain the driving factors behind each prediction using **SHAP (SHapley Additive exPlanations)**, and optimize decision thresholds based on actual business retention economics.

Built with **Python 3.12**, **Scikit-Learn**, **XGBoost**, **SHAP**, **FastAPI**, and a responsive **HTML5/CSS3/JavaScript (Chart.js)** frontend. Ready for local execution and seamless **Vercel** serverless deployment.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Machine Learning Workflow](#-machine-learning-workflow)
  - [1. Data Cleaning & EDA](#1-data-cleaning--eda)
  - [2. Preprocessing & Leakage Prevention](#2-preprocessing--leakage-prevention)
  - [3. Class Imbalance: Class Weights vs. SMOTE](#3-class-imbalance-class-weights-vs-smote)
  - [4. Model Comparison Benchmark](#4-model-comparison-benchmark)
  - [5. Cost-Sensitive Threshold Tuning](#5-cost-sensitive-threshold-tuning)
  - [6. Model Explainability with SHAP](#6-model-explainability-with-shap)
- [Web Application Architecture](#-web-application-architecture)
  - [FastAPI Backend API](#fastapi-backend-api)
  - [Frontend UI & Visualizations](#frontend-ui--visualizations)
  - [Zero-Downtime Client-Side Scoring Fallback](#zero-downtime-client-side-scoring-fallback)
- [Quick Start Guide](#-quick-start-guide)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
  - [Retraining the Models](#retraining-the-models)
  - [Running the Local Web App](#running-the-local-web-app)
- [API Reference](#-api-reference)
- [Vercel Deployment Guide](#-vercel-deployment-guide)

---

## 🎯 Project Overview

Telecommunications providers face substantial revenue churn when subscribers defect to competitors. Identifying at-risk customers *before* they leave—and understanding *why*—allows retention teams to offer targeted incentives to preserve Customer Lifetime Value (LTV).

### Key Highlights:
- **Dataset:** IBM Telco Customer Churn (7,043 customers, 21 attributes).
- **Champion Model:** Tuned **XGBoost Classifier** achieving **0.8482 ROC-AUC**.
- **Financial Optimization:** Cost-sensitive threshold tuning minimizes combined false-negative (lost customer LTV) and false-positive (unneeded discounts) costs, reducing retention program losses by **38.6%** ($37,300 saved on test cohort).
- **Explainability:** Fast local SHAP attribution via `TreeExplainer` delivers the top positive and negative churn drivers for each individual customer.
- **Dual-Mode Deployment:** Full-featured FastAPI + XGBoost backend with automated client-side Logistic Regression fallback using exported model coefficients and the sigmoid function.

---

## 📂 Repository Structure

```
churn-prediction-app/
├── notebooks/                              # Analysis, EDA, and modeling
│   ├── churn_analysis_and_modeling.ipynb   # Interactive Jupyter Notebook
│   ├── train_and_export.py                 # Automated training & artifact export pipeline
│   ├── build_notebook.py                   # Script to regenerate notebook
│   └── charts/                             # Saved EDA and evaluation figures
│       ├── churn_by_contract.png
│       ├── churn_by_tenure.png
│       ├── churn_by_monthly_charges.png
│       ├── churn_by_internet_service.png
│       ├── threshold_cost_curve.png
│       ├── roc_curve.png
│       ├── confusion_matrix_optimal.png
│       └── shap_global_bar.png
├── backend/                                # FastAPI server
│   ├── main.py                             # API application & endpoints
│   ├── model.joblib                        # Serialized Pipeline (ColumnTransformer + XGBoost)
│   ├── model_artifacts.joblib              # Explainer metadata, threshold, feature names
│   └── model_info.json                     # Model metrics & validation export
├── frontend/                               # Interactive Web UI
│   ├── index.html                          # Semantic form, gauge, and dashboard
│   ├── style.css                           # Modern responsive styling & animations
│   └── script.js                           # Form handling, Chart.js gauge, API integration
├── data/                                   # Data directory
│   ├── Telco-Customer-Churn.csv            # Raw dataset
│   ├── model_info.json                     # Model metrics export for frontend
│   └── customer_examples.json              # Pre-configured demo profiles
├── api/                                    # Vercel serverless function entrypoint
│   └── index.py                            # Exposes FastAPI app to Vercel
├── .gitignore                              # Git exclusion rules (venv, caches, etc.)
├── requirements.txt                        # Production runtime dependencies
├── requirements-dev.txt                    # Development requirements
├── vercel.json                             # Vercel serverless routing configuration
└── README.md                               # Project documentation
```

---

## 🔬 Machine Learning Workflow

### 1. Data Cleaning & EDA
- **TotalCharges Cleaning:** `TotalCharges` originally loads as string `object` due to 11 new customers (`tenure == 0`) having whitespace blanks (`" "`). Blanks are converted to numeric and imputed with `0.0`.
- **Target Distribution:** Overall churn rate is **26.54%** (5,174 retained, 1,869 churned), exhibiting moderate class imbalance.
- **Key EDA Findings:**
  1. **Contract Type:** Month-to-month contracts have an alarming **42.7%** churn rate, compared to **11.3%** for 1-year and **2.8%** for 2-year contracts.
  2. **Tenure:** Churn is heavily concentrated in the first 1–12 months. Customers past 24–36 months show strong stickiness.
  3. **Monthly Charges:** Churn peaks sharply between $70–$100/month (price-sensitive segment).
  4. **Internet Service:** Fiber optic subscribers churn at **41.9%**, vs. **19.0%** for DSL and **7.4%** for customers with no internet.

### 2. Preprocessing & Leakage Prevention
To ensure zero data leakage between train and test distributions:
- Used **Stratified 80/20 Train/Test Split** preserving the 26.54% churn ratio.
- Preprocessing encapsulated inside Scikit-Learn `ColumnTransformer`:
  - **Numeric Features (`tenure`, `MonthlyCharges`, `TotalCharges`):** Scaled via `StandardScaler()`.
  - **Categorical Features (16 columns):** Encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- Preprocessing steps fit strictly on training folds within pipelines.

### 3. Class Imbalance: Class Weights vs. SMOTE
We systematically compared cost-sensitive **Class Weights** (`class_weight='balanced'`, `scale_pos_weight = 2.769`) against **Synthetic Minority Over-sampling Technique (SMOTE)**:
- **Result:** Class weighting achieved comparable or higher ROC-AUC and F1 scores without generating synthetic interpolated points in one-hot spaces.

### 4. Model Comparison Benchmark

Evaluated on the independent stratified test set ($n = 1,409$):

| Model Architecture | Imbalance Strategy | Accuracy | ROC-AUC | F1-Score | Precision | Recall | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **Optimized Hyperparams** | **81.12%** | **0.8482** | **0.5994** | **68.62%** | **53.21%** | 🏆 **Champion** |
| Random Forest | Balanced Class Weights | 75.09% | 0.8428 | 0.6294 | 52.01% | 79.68% | Challenger |
| XGBoost | Balanced Class Weights | 75.37% | 0.8427 | 0.6305 | 52.39% | 79.14% | Challenger |
| XGBoost | SMOTE Resampling | 77.93% | 0.8422 | 0.6165 | 57.21% | 66.84% | Challenger |
| Logistic Regression | Baseline (Default) | 80.55% | 0.8420 | 0.6040 | 65.72% | 55.88% | Baseline |
| Logistic Regression | Balanced Class Weights | 73.81% | 0.8415 | 0.6136 | 50.43% | 78.34% | Challenger |
| Logistic Regression | SMOTE Resampling | 73.74% | 0.8399 | 0.6170 | 50.34% | 79.68% | Challenger |
| Random Forest | SMOTE Resampling | 76.44% | 0.8384 | 0.6295 | 54.02% | 75.40% | Challenger |

### 5. Cost-Sensitive Threshold Tuning
In churn management:
- **Cost of False Negative ($C_{FN}$):** $500 (Customer leaves undetected; loss of lifetime gross margin).
- **Cost of False Positive ($C_{FP}$):** $100 (Customer stays anyway; cost of unnecessary retention promotion).
- **Total Business Cost Function:** $\text{Cost}(T) = FN \times \$500 + FP \times \$100$.

By evaluating across thresholds $T \in [0.10, 0.90]$:
- **Default Threshold (0.50):** Cost = **$96,600.00**
- **Optimal Threshold (0.15):** Cost = **$59,300.00**
- **Financial Savings:** **$37,300.00 (38.6% cost reduction)**.

### 6. Model Explainability with SHAP
Using `shap.TreeExplainer(xgb_clf)`:
- **Global Top Drivers:** Month-to-month contracts, low customer tenure, no online security, high monthly charges, and fiber optic connection.
- **Local Explanations:** For each incoming prediction request, SHAP values are converted to human-readable explanations:
  - *Example:* `"Month-to-month contract strongly increases churn risk (no long-term commitment)."`
  - *Example:* `"Established loyalty (65 months tenure) substantially reduces churn risk."`

---

## 💻 Web Application Architecture

### FastAPI Backend API
- **`POST /predict`**: Accepts customer profile JSON, returns churn probability, risk classification (`Low`, `Medium`, `High`), cost-tuned decision, and top 4 SHAP driver explanations.
- **`GET /model-info`**: Returns model metrics, ROC curve data points, global feature importance, and multi-model benchmark statistics.
- **`GET /customer-examples`**: Provides preset customer profiles for rapid demonstrations.
- **`GET /health`**: Returns system health, loaded models, and feature counts.

### Frontend UI & Visualizations
- **Interactive Form:** Sliders with dynamic value sync, radio pills, and dependent dropdowns.
- **Semi-Circular Gauge:** Animated Chart.js doughnut gauge rendering predicted churn probability.
- **Key Driver Cards:** Colored indicators (🔼 red for risk increase, 🔽 green for risk reduction) detailing exact SHAP impact.
- **Benchmark Dashboard:** Interactive Chart.js horizontal bar chart for feature importances, ROC curve line chart, and model comparison table.
- **Preset Selector:** One-click demo buttons:
  - *Jane Doe:* High Risk Churner (Month-to-month, fiber optic, electronic check, 2-month tenure).
  - *Marcus Vance:* Loyal Long-term Customer (Two-year contract, 65-month tenure, full security).
  - *Alex Chen:* Moderate Risk Customer (1-year contract, 18-month tenure).
  - *Robert Miller:* Budget Saver (Phone-only, no internet).

### Zero-Downtime Client-Side Scoring Fallback
If the frontend is served statically without an active backend server (or during offline development):
- JavaScript computes the churn probability locally using the exported baseline Logistic Regression weights:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \beta_0 + \sum_{i} \beta_i x_i$$
- Individual coefficient contributions are sorted to provide instant explainability.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Virtual environment (`venv`)

### Installation & Setup

1. **Clone the repository and enter directory:**
   ```bash
   cd /home/sharmajisumit/Projects/churn-prediction-app
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Retraining the Models
To run the full end-to-end training, generate charts, compute SHAP values, and export model artifacts:
```bash
python notebooks/train_and_export.py
```

To explore or execute the interactive Jupyter notebook:
```bash
jupyter notebook notebooks/churn_analysis_and_modeling.ipynb
```

### Running the Local Web App
Run the FastAPI application with Uvicorn:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser to:
- **Interactive Web App:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **FastAPI Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 API Reference

### `POST /predict`
**Request Body:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
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
  "TotalCharges": 179.0
}
```

**Response (200 OK):**
```json
{
  "probability": 0.7682,
  "risk_level": "High",
  "optimal_threshold": 0.15,
  "predicted_churn": true,
  "top_reasons": [
    {
      "feature": "Contract: Month-to-month",
      "impact": 0.661,
      "direction": "increase",
      "human_explanation": "Month-to-month contract strongly increases churn risk (no long-term commitment)."
    },
    {
      "feature": "tenure",
      "impact": 0.576,
      "direction": "increase",
      "human_explanation": "Short tenure (2 months) increases churn risk during onboarding."
    },
    {
      "feature": "OnlineSecurity: No",
      "impact": 0.223,
      "direction": "increase",
      "human_explanation": "No online security add-on increases customer churn likelihood."
    },
    {
      "feature": "MonthlyCharges",
      "impact": 0.184,
      "direction": "increase",
      "human_explanation": "Higher monthly bill ($89.50) elevates price sensitivity and churn risk."
    }
  ]
}
```

---

## ☁️ Vercel Deployment Guide

This project is pre-configured for Vercel Serverless deployment using [`vercel.json`](vercel.json) and [`api/index.py`](api/index.py).

### Steps to Deploy:
1. **Push the repository to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Telco churn prediction web app"
   git remote add origin https://github.com/<your-username>/churn-prediction-app.git
   git push -u origin main
   ```

2. **Import into Vercel:**
   - Log in to [vercel.com](https://vercel.com).
   - Click **Add New... > Project** and select your GitHub repository.
   - Framework Preset: **Other**.
   - Root Directory: `./`.
   - Click **Deploy**.

3. **How Vercel handles the application:**
   - Vercel automatically detects `api/index.py` and runs it as a Python Serverless Function.
   - `vercel.json` routes `/predict`, `/model-info`, `/customer-examples`, and `/health` directly to the FastAPI serverless function.
   - All static assets in `/frontend` (`index.html`, `style.css`, `script.js`) are served globally via Vercel's Edge CDN.
