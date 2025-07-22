import pandas as pd
import numpy as np

# Load the dataset

# Use CSV file for faster processing
file_path = 'Delinquency_prediction_dataset.csv'
df = pd.read_csv(file_path)

# 1. Impute numerical columns with median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    median = df[col].median()
    df[col].fillna(median, inplace=True)

# 2. Impute categorical columns with mode, or 'Missing' if mode is not available
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    if df[col].isnull().any():
        mode = df[col].mode()
        if not mode.empty:
            df[col].fillna(mode[0], inplace=True)
        else:
            df[col].fillna('Missing', inplace=True)

# 3. Add missing value indicator columns
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[f'{col}_was_missing'] = df[col].isnull().astype(int)

# 4. Save the updated dataset
output_path = 'Delinquency_prediction_dataset_imputed.csv'
df.to_csv(output_path, index=False)
print(f'Imputation complete. Updated dataset saved to {output_path}')
