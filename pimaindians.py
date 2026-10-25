import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. Load and Preprocess the Dataset
# ==========================================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Replace biological zero-placeholders with NaN
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Impute missing values with the median
imputer = SimpleImputer(strategy='median')
df[cols_with_zeros] = imputer.fit_transform(df[cols_with_zeros])

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split into Training (80%) and Testing (20%) sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 2. Model 1: Logistic Regression WITHOUT Scaling
# ==========================================
lr_unscaled = LogisticRegression(max_iter=1000, random_state=42)
lr_unscaled.fit(X_train, y_train)
y_pred_unscaled = lr_unscaled.predict(X_test)

# ==========================================
# 3. Model 2: Logistic Regression WITH Scaling
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_scaled = LogisticRegression(max_iter=1000, random_state=42)
lr_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = lr_scaled.predict(X_test_scaled)

# ==========================================
# 4. Model Evaluation & Comparison
# ==========================================
def compute_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }

metrics_unscaled = compute_metrics(y_test, y_pred_unscaled)
metrics_scaled = compute_metrics(y_test, y_pred_scaled)

results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Without Scaling': [metrics_unscaled['Accuracy'], metrics_unscaled['Precision'], metrics_unscaled['Recall'], metrics_unscaled['F1-Score']],
    'With Scaling': [metrics_scaled['Accuracy'], metrics_scaled['Precision'], metrics_scaled['Recall'], metrics_scaled['F1-Score']]
})

print("\n=== Performance Metrics Comparison ===")
print(results_df.to_string(index=False))

# ==========================================
# 5. Coefficient Analysis
# ==========================================
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Unscaled Coef': lr_unscaled.coef_[0],
    'Scaled Coef': lr_scaled.coef_[0]
})

print("\n=== Feature Coefficients Comparison ===")
print(coef_df.to_string(index=False))
