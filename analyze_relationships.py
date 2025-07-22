import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the imputed dataset
file_path = 'Delinquency_prediction_dataset_imputed.csv'
df = pd.read_csv(file_path)

# Assume the target variable is named 'delinquency' (1 = delinquent, 0 = not delinquent)
# If the actual column name is different, please update accordingly.

def analyze_relationships(df):
    results = []
    # Example: Credit Utilization vs Delinquency
    if 'credit_utilization' in df.columns and 'delinquency' in df.columns:
        avg_util_delinquent = df[df['delinquency'] == 1]['credit_utilization'].mean()
        avg_util_non = df[df['delinquency'] == 0]['credit_utilization'].mean()
        results.append(f"Average credit utilization for delinquent accounts: {avg_util_delinquent:.2f}")
        results.append(f"Average credit utilization for non-delinquent accounts: {avg_util_non:.2f}")
    # Example: Payment History vs Delinquency
    if 'payment_history' in df.columns and 'delinquency' in df.columns:
        avg_payhist_delinquent = df[df['delinquency'] == 1]['payment_history'].mean()
        avg_payhist_non = df[df['delinquency'] == 0]['payment_history'].mean()
        results.append(f"Average payment history score for delinquent accounts: {avg_payhist_delinquent:.2f}")
        results.append(f"Average payment history score for non-delinquent accounts: {avg_payhist_non:.2f}")
    # Example: Debt-to-Income Ratio vs Delinquency
    if 'dti' in df.columns and 'delinquency' in df.columns:
        avg_dti_delinquent = df[df['delinquency'] == 1]['dti'].mean()
        avg_dti_non = df[df['delinquency'] == 0]['dti'].mean()
        results.append(f"Average DTI for delinquent accounts: {avg_dti_delinquent:.2f}")
        results.append(f"Average DTI for non-delinquent accounts: {avg_dti_non:.2f}")
    # Correlation heatmap
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    results.append('Correlation heatmap saved as correlation_heatmap.png')
    return results

if __name__ == "__main__":
    summary = analyze_relationships(df)
    for line in summary:
        print(line)
