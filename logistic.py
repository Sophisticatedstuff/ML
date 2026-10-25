import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# ==========================================
# 1. Load and Preprocess the Dataset
# ==========================================
print("Loading Breast Cancer Wisconsin dataset...")
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

# Split into Training (80%) and Testing (20%) sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling (Essential for regularized optimization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. MLE & MAP Estimation Models
# ==========================================
# A. MLE: Logistic Regression with NO penalty (penalty=None)
mle_model = LogisticRegression(penalty=None, max_iter=10000, random_state=42)
mle_model.fit(X_train_scaled, y_train)

# B. MAP with L2 Regularization (Gaussian Prior, C = 1.0)
map_l2_model = LogisticRegression(penalty='l2', C=1.0, max_iter=10000, random_state=42)
map_l2_model.fit(X_train_scaled, y_train)

# C. MAP with L1 Regularization (Laplace Prior, C = 1.0)
map_l1_model = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=10000, random_state=42)
map_l1_model.fit(X_train_scaled, y_train)

# ==========================================
# 3. Model Evaluation Comparison
# ==========================================
models = {
    'MLE (Unregularized)': mle_model,
    'MAP - L2 (Gaussian Prior)': map_l2_model,
    'MAP - L1 (Laplace Prior)': map_l1_model
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)
    
    results.append({
        'Model': name,
        'Test Accuracy': f"{acc * 100:.2f}%",
        'Log Loss': f"{loss:.4f}"
    })

results_df = pd.DataFrame(results)
print("\n=== Performance Metrics Comparison ===")
print(results_df.to_string(index=False))

# ==========================================
# 4. Parameter (Coefficient) Comparison
# ==========================================
coef_df = pd.DataFrame({
    'Feature': data.feature_names,
    'MLE Coef': mle_model.coef_[0],
    'MAP (L2) Coef': map_l2_model.coef_[0],
    'MAP (L1) Coef': map_l1_model.coef_[0]
})

print("\n=== Sample Parameter Estimates (First 10 Features) ===")
print(coef_df.head(10).to_string(index=False))
