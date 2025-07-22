
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load the imputed dataset
df = pd.read_csv('Delinquency_prediction_dataset_imputed.csv')


# Print all column names to help identify correct feature names
print('Available columns in the dataset:')
print(list(df.columns))




# Encode categorical/text columns to numeric (robust to case, hyphens, spaces)
encoding_map = {
    'ontime': 0, 'on-time': 0, 'on time': 0,
    'late': 1,
    'missed': 2
}
def normalize_str(val):
    if isinstance(val, str):
        return val.strip().lower().replace('-', '').replace(' ', '')
    return val
for col in ['Missed_Payments', 'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']:
    if col in df.columns:
        df[col] = df[col].apply(normalize_str).map(encoding_map).fillna(df[col]).astype(float)

# Select relevant features and target variable (expanded feature set)
features = [
    'Income', 'Credit_Utilization', 'Missed_Payments', 'Debt_to_Income_Ratio', 'Credit_Score',
    'Age', 'Loan_Balance', 'Account_Tenure',
    'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6'
]
target = 'Delinquent_Account'
X = df[features]
y = df[target]


# Preprocessing: scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)




# Model 1: Logistic Regression (on SMOTE data)
logreg = LogisticRegression()
logreg.fit(X_train_res, y_train_res)
y_pred_logreg = logreg.predict(X_test)
print('--- Logistic Regression (SMOTE) Results ---')
print(classification_report(y_test, y_pred_logreg))
print('ROC-AUC:', roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1]))


# Model 2: XGBoost Classifier (on SMOTE data) with hyperparameter tuning and feature importance
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Hyperparameter grid for XGBoost
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}

xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
rs = RandomizedSearchCV(xgb_base, param_distributions=param_dist, n_iter=20, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1, verbose=1)
rs.fit(X_train_res, y_train_res)
xgb_best = rs.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)
print('\n--- XGBoost (SMOTE, Tuned) Results ---')
print('Best XGBoost Params:', rs.best_params_)
print(classification_report(y_test, y_pred_xgb))
print('ROC-AUC:', roc_auc_score(y_test, xgb_best.predict_proba(X_test)[:, 1]))

# Feature importance
importances = xgb_best.feature_importances_
feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
print('\nXGBoost Feature Importances:')
for feat, score in feature_importance:
    print(f'{feat}: {score:.4f}')

# Example: Predict risk for a new customer (replace with actual values)
# new_customer = pd.DataFrame({
#     'credit_utilization': [0.5],
#     'payment_history': [0.8],
#     'dti': [0.35],
#     'income': [50000],
#     'recent_credit_inquiries': [2]
# })
# new_customer_scaled = scaler.transform(new_customer)
# risk_score = model.predict_proba(new_customer_scaled)[:, 1]
# print('Predicted delinquency risk:', risk_score)
