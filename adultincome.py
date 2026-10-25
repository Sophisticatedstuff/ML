import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ==========================================
# 1. Load and Preprocess the Dataset
# ==========================================
print("Loading Adult Income dataset from OpenML...")
adult = fetch_openml('adult', version=2, as_frame=True, parser='auto')
df = adult.frame

# Clean target variable names (strip whitespace and map to binary 0/1)
df['class'] = df['class'].astype(str).str.strip().map({'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K.': 1})

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['category', 'object']).columns.tolist()

# Split into Training (80%) and Testing (20%) sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipelines for numerical (scaling) and categorical (one-hot encoding) data
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ==========================================
# 2. Implement Models using Pipelines
# ==========================================

# A. Logistic Regression Pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# B. Decision Tree Pipeline
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=10, random_state=42))
])

print("Training Logistic Regression...")
lr_pipeline.fit(X_train, y_train)

print("Training Decision Tree...")
dt_pipeline.fit(X_train, y_train)

# ==========================================
# 3. Model Evaluation & Comparison
# ==========================================
y_pred_lr = lr_pipeline.predict(X_test)
y_pred_dt = dt_pipeline.predict(X_test)

def evaluate_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }

metrics_lr = evaluate_metrics(y_test, y_pred_lr)
metrics_dt = evaluate_metrics(y_test, y_pred_dt)

results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Logistic Regression': [metrics_lr['Accuracy'], metrics_lr['Precision'], metrics_lr['Recall'], metrics_lr['F1-Score']],
    'Decision Tree': [metrics_dt['Accuracy'], metrics_dt['Precision'], metrics_dt['Recall'], metrics_dt['F1-Score']]
})

print("\n=== Performance Metrics Comparison ===")
print(results_df.to_string(index=False))
