# Delinquency Prediction Project

## Overview
This project uses machine learning models to predict the likelihood of account delinquency based on customer financial and behavioral data. The workflow includes data preprocessing, feature engineering, model training, evaluation, and interpretation. Both Logistic Regression and XGBoost models are implemented, with SMOTE applied to address class imbalance.

## Features Used
- Income
- Credit Utilization
- Missed Payments
- Debt-to-Income Ratio
- Credit Score
- Age
- Loan Balance
- Account Tenure
- Payment history for the last 6 months (Month_1 to Month_6)

## Workflow
1. **Data Preprocessing:**
   - Load and clean the dataset
   - Encode categorical variables
   - Standardize features
2. **Train-Test Split:**
   - Split data into training and testing sets
3. **SMOTE:**
   - Apply SMOTE to balance the training data
4. **Model Training:**
   - Train Logistic Regression and XGBoost models
   - Hyperparameter tuning for XGBoost using RandomizedSearchCV
5. **Evaluation:**
   - Evaluate models using classification metrics (accuracy, precision, recall, F1, ROC-AUC)
   - Analyze feature importance

## Key Insights
- Missed payments and high credit utilization are the strongest predictors of delinquency.
- Customers with multiple recent missed payments and high utilization are at the highest risk.
- XGBoost provides superior predictive performance, while Logistic Regression offers interpretability.

## Usage
1. Ensure all dependencies are installed (see below).
2. Place the imputed dataset (`Delinquency_prediction_dataset_imputed.csv`) in the project directory.
3. Run `model_logic.py` to train models and view results.

## Dependencies
- pandas
- scikit-learn
- xgboost
- imbalanced-learn

Install dependencies with:
```
pip install pandas scikit-learn xgboost imbalanced-learn
```

## Ethical Considerations
- The models are regularly audited for bias and fairness.
- Explanations for predictions are provided to support transparency and compliance.
- Customer data privacy and security are maintained throughout the process.

## Contact
For questions or support, contact Arya Prakash Shrivastav at aryaprakash1759@gmail.com.
