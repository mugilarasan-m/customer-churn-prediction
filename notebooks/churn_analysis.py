import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import load_and_clean_data

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style='whitegrid')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(project_root, 'data', 'customer_churn.csv')

df = load_and_clean_data(data_path)
print("Dataset Summary:")
print(df.info())
print(df.describe())

plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title('Churn Class Distribution (0 = Retained, 1 = Churned)')
plt.savefig('churn_distribution.png')
plt.close()

cat_cols_to_plot = ['Contract', 'InternetService', 'PaymentMethod', 'SeniorCitizen']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, col in enumerate(cat_cols_to_plot):
    row, c = divmod(idx, 2)
    sns.countplot(x=col, hue='Churn', data=df, ax=axes[row, c], palette='Set1')
    axes[row, c].set_title(f'Churn by {col}')
    axes[row, c].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('categorical_churn_analysis.png')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

for idx, col in enumerate(num_cols):
    sns.kdeplot(data=df, x=col, hue='Churn', common_norm=False, fill=True, ax=axes[idx], palette='tab10')
    axes[idx].set_title(f'{col} Distribution by Churn Status')

plt.tight_layout()
plt.savefig('numerical_churn_analysis.png')
plt.close()

print("EDA plots successfully generated and saved to root.")