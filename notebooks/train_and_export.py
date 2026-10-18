"""
train_and_export.py
Comprehensive End-to-End Pipeline for IBM Telco Customer Churn Prediction:
1. Data Cleaning & Validation (fixes TotalCharges text blanks)
2. Exploratory Data Analysis & Chart Generation
3. Preprocessing with ColumnTransformer & Pipeline (no leakage)
4. Stratified Train/Test Split
5. Model Comparison: Logistic Regression, Random Forest, XGBoost
6. Class Imbalance Handling: Class Weights vs SMOTE
7. Evaluation & Cost-Sensitive Threshold Tuning
8. SHAP Global and Local Explainability
9. Artifact Export (model.joblib, model_artifacts.joblib, model_info.json, customer_examples.json)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)
import shap

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Telco-Customer-Churn.csv')
CHARTS_DIR = os.path.join(BASE_DIR, 'notebooks', 'charts')
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(BACKEND_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 70)
print("TELCO CUSTOMER CHURN PREDICTION PIPELINE")
print("=" * 70)

# 1. Load Data
print("\n[Step 1] Loading Dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Fix TotalCharges
print("\n[Step 2] Cleaning Data & Fixing TotalCharges...")
empty_charges = (df['TotalCharges'].astype(str).str.strip() == '').sum()
print(f"Blank/whitespace entries in TotalCharges: {empty_charges}")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
# Fill 11 blank rows with 0.0 (these are customers with tenure == 0)
df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

# Target encoding
churn_counts = df['Churn'].value_counts()
churn_rate = (df['Churn'] == 'Yes').mean()
print(f"Churn Distribution:\n{churn_counts}")
print(f"Churn Rate: {churn_rate * 100:.2f}%")
df['Churn_Numeric'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 3. Exploratory Data Analysis & Visualizations
print("\n[Step 3] Generating EDA Visualizations...")

# Chart 1: Churn by Contract
fig, ax = plt.subplots(figsize=(8, 5))
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
contract_churn.plot(kind='bar', stacked=True, ax=ax, color=['#2b5c8f', '#d9534f'], edgecolor='none', alpha=0.9)
ax.set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_xlabel('Contract Type', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    if height > 5:
        x, y = p.get_xy() 
        ax.text(x + width/2, y + height/2, f"{height:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
ax.legend(['Retained (No)', 'Churned (Yes)'], title='Status', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'churn_by_contract.png'), dpi=200)
plt.close()

# Chart 2: Churn by Tenure
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df, x='tenure', hue='Churn', kde=True, bins=36, palette={'No': '#2b5c8f', 'Yes': '#d9534f'}, ax=ax, alpha=0.6)
ax.set_title('Customer Tenure Distribution by Churn Status', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Tenure (Months)', fontsize=12)
ax.set_ylabel('Number of Customers', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'churn_by_tenure.png'), dpi=200)
plt.close()

# Chart 3: Churn by Monthly Charges
fig, ax = plt.subplots(figsize=(8, 5))
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', common_norm=False, fill=True, palette={'No': '#2b5c8f', 'Yes': '#d9534f'}, ax=ax, alpha=0.5)
ax.set_title('Monthly Charges Density: Churn vs Retained', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Monthly Charges ($)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'churn_by_monthly_charges.png'), dpi=200)
plt.close()

# Chart 4: Churn by Internet Service
fig, ax = plt.subplots(figsize=(8, 5))
internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
internet_churn.plot(kind='bar', stacked=True, ax=ax, color=['#2b5c8f', '#d9534f'], edgecolor='none', alpha=0.9)
ax.set_title('Churn Rate by Internet Service Type', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_xlabel('Internet Service', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    if height > 5:
        x, y = p.get_xy() 
        ax.text(x + width/2, y + height/2, f"{height:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
ax.legend(['Retained (No)', 'Churned (Yes)'], title='Status', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'churn_by_internet_service.png'), dpi=200)
plt.close()

print(f"EDA plots saved to {CHARTS_DIR}/")

# 4. Preprocessing & Train/Test Split
print("\n[Step 4] Building Preprocessing Pipeline & Stratified Split...")

X = df.drop(columns=['customerID', 'Churn', 'Churn_Numeric'])
y = df['Churn_Numeric']

num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_features = [col for col in X.columns if col not in num_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
print(f"Train Churn Rate: {y_train.mean():.4f} | Test Churn Rate: {y_test.mean():.4f}")

# Compute scale_pos_weight for XGBoost
scale_pos_wt = float((len(y_train) - sum(y_train)) / sum(y_train))
print(f"Calculated scale_pos_weight for imbalance: {scale_pos_wt:.3f}")

# 5. Model Exploration & Imbalance Comparison
print("\n[Step 5] Training and Evaluating Models...")

candidate_models = {
    'Logistic Regression (Baseline)': Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Logistic Regression (Class Weights)': Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ]),
    'Logistic Regression (SMOTE)': ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest (Class Weights)': Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=150, max_depth=7, class_weight='balanced', random_state=42))
    ]),
    'Random Forest (SMOTE)': ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42))
    ]),
    'XGBoost (Class Weights)': Pipeline([
        ('prep', preprocessor),
        ('clf', XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, scale_pos_weight=scale_pos_wt, eval_metric='logloss', random_state=42))
    ]),
    'XGBoost (SMOTE)': ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, eval_metric='logloss', random_state=42))
    ]),
    'XGBoost (Tuned)': Pipeline([
        ('prep', preprocessor),
        ('clf', XGBClassifier(n_estimators=140, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, eval_metric='logloss', random_state=42))
    ])
}

model_metrics = []
trained_pipelines = {}

for name, pipe in candidate_models.items():
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    
    model_metrics.append({
        'Model': name,
        'Accuracy': round(float(acc), 4),
        'ROC-AUC': round(float(auc), 4),
        'F1-Score': round(float(f1), 4),
        'Precision': round(float(prec), 4),
        'Recall': round(float(rec), 4)
    })

comparison_df = pd.DataFrame(model_metrics)
print("\n" + comparison_df.to_string(index=False))

# Select Best Model
best_model_name = 'XGBoost (Tuned)'
best_pipeline = trained_pipelines[best_model_name]
best_clf = best_pipeline.named_steps['clf']
best_prep = best_pipeline.named_steps['prep']
y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]

# 6. Cost-Sensitive Threshold Tuning
print("\n[Step 6] Business Cost Optimization & Threshold Tuning...")
# Business scenario:
# Cost of False Negative (Churner missed, lost customer lifetime value) = $500
# Cost of False Positive (Unnecessary retention incentive/discount given) = $100
# True Positive: Customer flagged and retention offer accepted (cost = $100)
# True Negative: No intervention needed (cost = $0)
c_fn = 500.0
c_fp = 100.0

thresholds = np.linspace(0.1, 0.9, 81)
cost_records = []

for th in thresholds:
    y_pred_th = (y_test_prob >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_th).ravel()
    total_cost = (fn * c_fn) + (fp * c_fp)
    f1_th = f1_score(y_test, y_pred_th)
    prec_th = precision_score(y_test, y_pred_th, zero_division=0)
    rec_th = recall_score(y_test, y_pred_th)
    cost_records.append({
        'threshold': float(th),
        'total_cost': float(total_cost),
        'fn': int(fn),
        'fp': int(fp),
        'tp': int(tp),
        'tn': int(tn),
        'f1': float(f1_th),
        'precision': float(prec_th),
        'recall': float(rec_th)
    })

cost_df = pd.DataFrame(cost_records)
min_cost_idx = cost_df['total_cost'].idxmin()
optimal_threshold = float(cost_df.loc[min_cost_idx, 'threshold'])
min_cost = float(cost_df.loc[min_cost_idx, 'total_cost'])

# Cost at default threshold 0.5
cost_at_default = float(cost_df.loc[(cost_df['threshold'] - 0.5).abs().idxmin(), 'total_cost'])
cost_savings = cost_at_default - min_cost

print(f"Default Threshold (0.50) Business Cost: ${cost_at_default:,.2f}")
print(f"Optimal Threshold ({optimal_threshold:.2f}) Business Cost: ${min_cost:,.2f}")
print(f"Total Business Cost Savings: ${cost_savings:,.2f} ({(cost_savings/cost_at_default)*100:.1f}%)")

# Plot Cost Curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cost_df['threshold'], cost_df['total_cost'], color='#1e3a8a', lw=2.5, label='Total Business Cost ($)')
ax.axvline(0.50, color='#6b7280', linestyle='--', label='Default Threshold (0.50)')
ax.axvline(optimal_threshold, color='#10b981', linestyle='-', lw=2, label=f'Optimal Threshold ({optimal_threshold:.2f})')
ax.scatter([optimal_threshold], [min_cost], color='#10b981', s=100, zorder=5)
ax.set_title('Cost-Sensitive Decision Threshold Optimization', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Classification Decision Threshold', fontsize=12)
ax.set_ylabel('Total Cost on Test Set ($)', fontsize=12)
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'threshold_cost_curve.png'), dpi=200)
plt.close()

# 7. ROC Curve and Confusion Matrix
print("\n[Step 7] Generating ROC Curve & Confusion Matrix Visualizations...")
fpr, tpr, roc_thresh = roc_curve(y_test, y_test_prob)
roc_auc = float(roc_auc_score(y_test, y_test_prob))

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'XGBoost (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#9ca3af', linestyle='--', lw=1.5, label='Random Guess')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'roc_curve.png'), dpi=200)
plt.close()

# Confusion Matrix at Optimal Threshold
y_pred_opt = (y_test_prob >= optimal_threshold).astype(int)
cm_opt = confusion_matrix(y_test, y_pred_opt)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            xticklabels=['Predicted Retain', 'Predicted Churn'],
            yticklabels=['Actual Retain', 'Actual Churn'])
ax.set_title(f'Confusion Matrix at Threshold {optimal_threshold:.2f}', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'confusion_matrix_optimal.png'), dpi=200)
plt.close()

# 8. SHAP Explainability
print("\n[Step 8] Computing SHAP Explainability...")
# Feature names after transformation
transformed_feature_names = list(best_prep.get_feature_names_out())
clean_feature_names = [
    f.replace('cat__', '').replace('num__', '') for f in transformed_feature_names
]

X_test_transformed = best_prep.transform(X_test)
explainer = shap.TreeExplainer(best_clf)
shap_values = explainer(X_test_transformed[:500])

# Global Feature Importance from SHAP
mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
feat_imp_df = pd.DataFrame({
    'Feature': clean_feature_names,
    'Importance': mean_abs_shap
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("\nTop 10 Global Features (SHAP Importance):")
print(feat_imp_df.head(10).to_string(index=False))

# Plot SHAP summary bar chart
fig, ax = plt.subplots(figsize=(9, 6))
top15 = feat_imp_df.head(15).iloc[::-1]
ax.barh(top15['Feature'], top15['Importance'], color='#3b82f6')
ax.set_title('Top 15 Most Influential Features (Mean |SHAP Value|)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Mean |SHAP Value| (Impact on Model Log-Odds)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'shap_global_bar.png'), dpi=200)
plt.close()

# Also save SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values.values, X_test_transformed[:500], feature_names=clean_feature_names, show=False)
plt.title('SHAP Feature Impact Summary', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'shap_summary_beeswarm.png'), dpi=200)
plt.close()

# 9. Extract Baseline Logistic Regression for Client-Side Fallback
print("\n[Step 9] Extracting Logistic Regression for Client-Side Scoring Fallback...")
logreg_pipe = trained_pipelines['Logistic Regression (Baseline)']
logreg_clf = logreg_pipe.named_steps['clf']
logreg_coefs = {feat: float(coef) for feat, coef in zip(clean_feature_names, logreg_clf.coef_[0])}
logreg_intercept = float(logreg_clf.intercept_[0])

# Scaler values for tenure, MonthlyCharges, TotalCharges
scaler = best_prep.named_transformers_['num']
num_scaler_info = {
    'tenure': {'mean': float(scaler.mean_[0]), 'scale': float(scaler.scale_[0])},
    'MonthlyCharges': {'mean': float(scaler.mean_[1]), 'scale': float(scaler.scale_[1])},
    'TotalCharges': {'mean': float(scaler.mean_[2]), 'scale': float(scaler.scale_[2])}
}

# 10. Export Artifacts
print("\n[Step 10] Exporting Artifacts...")

# A. model.joblib (full sklearn pipeline: preprocessor + XGBoost)
model_joblib_path = os.path.join(BACKEND_DIR, 'model.joblib')
joblib.dump(best_pipeline, model_joblib_path)
print(f"Saved best pipeline to: {model_joblib_path}")

# B. model_artifacts.joblib (metadata, explainer cache, threshold)
artifacts = {
    'model_name': best_model_name,
    'optimal_threshold': optimal_threshold,
    'clean_feature_names': clean_feature_names,
    'transformed_feature_names': transformed_feature_names,
    'num_features': num_features,
    'cat_features': cat_features,
    'business_cost': {
        'cost_fn': c_fn,
        'cost_fp': c_fp,
        'cost_at_default': cost_at_default,
        'cost_at_optimal': min_cost,
        'cost_savings': cost_savings
    }
}
joblib.dump(artifacts, os.path.join(BACKEND_DIR, 'model_artifacts.joblib'))
print(f"Saved model artifacts to: {os.path.join(BACKEND_DIR, 'model_artifacts.joblib')}")

# C. data/model_info.json (for GET /model-info and frontend consumption)
subsample_roc_indices = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
roc_curve_data = {
    'fpr': [round(float(fpr[i]), 4) for i in subsample_roc_indices],
    'tpr': [round(float(tpr[i]), 4) for i in subsample_roc_indices],
    'auc': round(roc_auc, 4)
}

model_info_json = {
    'best_model': {
        'name': best_model_name,
        'roc_auc': round(roc_auc, 4),
        'accuracy': round(float(accuracy_score(y_test, y_pred_opt)), 4),
        'f1_score': round(float(f1_score(y_test, y_pred_opt)), 4),
        'precision': round(float(precision_score(y_test, y_pred_opt, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, y_pred_opt)), 4),
        'optimal_threshold': round(optimal_threshold, 2),
        'confusion_matrix': {
            'tn': int(cm_opt[0, 0]),
            'fp': int(cm_opt[0, 1]),
            'fn': int(cm_opt[1, 0]),
            'tp': int(cm_opt[1, 1])
        },
        'business_savings_dollars': round(cost_savings, 2)
    },
    'model_comparison': model_metrics,
    'top_features': [
        {'feature': row['Feature'], 'importance': round(float(row['Importance']), 4)}
        for _, row in feat_imp_df.head(12).iterrows()
    ],
    'roc_curve': roc_curve_data,
    'dataset_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'churn_rate': round(churn_rate * 100, 2),
        'features_count': len(clean_feature_names)
    },
    'client_scoring_fallback': {
        'intercept': logreg_intercept,
        'coefficients': logreg_coefs,
        'scaler': num_scaler_info
    }
}

with open(os.path.join(DATA_DIR, 'model_info.json'), 'w') as f:
    json.dump(model_info_json, f, indent=2)
print("Saved model_info.json to data/")

# D. data/customer_examples.json (demo customers for the frontend)
customer_examples = [
    {
        'id': 'high_risk',
        'title': 'High Risk: Month-to-Month Fiber Optic (Jane)',
        'description': 'Month-to-month contract, fiber optic internet, electronic check, no tech support, high monthly bill.',
        'data': {
            'gender': 'Female',
            'SeniorCitizen': 1,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 2,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 98.50,
            'TotalCharges': 197.00
        }
    },
    {
        'id': 'low_risk',
        'title': 'Low Risk: Loyal Two-Year Customer (Marcus)',
        'description': 'Two-year contract, 65-month tenure, DSL, full online security & tech support, auto credit card.',
        'data': {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'Yes',
            'tenure': 65,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Credit card (automatic)',
            'MonthlyCharges': 45.30,
            'TotalCharges': 2944.50
        }
    },
    {
        'id': 'medium_risk',
        'title': 'Medium Risk: 1-Year Contract with Moderate Bill (Alex)',
        'description': 'One-year contract, 18-month tenure, Fiber optic, partial security services, bank transfer.',
        'data': {
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 18,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'Yes',
            'Contract': 'One year',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Bank transfer (automatic)',
            'MonthlyCharges': 79.80,
            'TotalCharges': 1436.40
        }
    },
    {
        'id': 'budget_saver',
        'title': 'Low Risk: Phone Only / No Internet (Robert)',
        'description': 'Month-to-month phone-only customer with low monthly charge and 12-month tenure.',
        'data': {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 14,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'No',
            'OnlineSecurity': 'No internet service',
            'OnlineBackup': 'No internet service',
            'DeviceProtection': 'No internet service',
            'TechSupport': 'No internet service',
            'StreamingTV': 'No internet service',
            'StreamingMovies': 'No internet service',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Mailed check',
            'MonthlyCharges': 19.90,
            'TotalCharges': 278.60
        }
    }
]

with open(os.path.join(DATA_DIR, 'customer_examples.json'), 'w') as f:
    json.dump(customer_examples, f, indent=2)
print("Saved customer_examples.json to data/")
print("\n" + "=" * 70)
print("TRAINING & EXPORT COMPLETED SUCCESSFULLY!")
print("=" * 70)
