import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. Load and Preprocess the Dataset
# ==========================================
print("Loading Diabetes dataset...")
diabetes = load_diabetes(as_frame=True)
X = diabetes.data
y = diabetes.target

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling (Crucial for regularized models so penalties apply equally)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. Implementations & Hyperparameter Tuning
# ==========================================

# A. Standard Linear Regression (OLS)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# B. Ridge Regression with Built-in Cross-Validation (RidgeCV)
alphas_ridge = np.logspace(-3, 4, 200)  # Test 200 alpha values from 0.001 to 10,000
ridge_cv = RidgeCV(alphas=alphas_ridge, scoring='neg_mean_squared_error', cv=5)
ridge_cv.fit(X_train_scaled, y_train)
ridge_pred = ridge_cv.predict(X_test_scaled)

# C. Lasso Regression with Built-in Cross-Validation (LassoCV)
lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
lasso_pred = lasso_cv.predict(X_test_scaled)

# ==========================================
# 3. Model Performance Evaluation
# ==========================================
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

lr_mse, lr_r2 = evaluate(y_test, lr_pred)
ridge_mse, ridge_r2 = evaluate(y_test, ridge_pred)
lasso_mse, lasso_r2 = evaluate(y_test, lasso_pred)

results = pd.DataFrame({
    'Model': ['Linear Regression (OLS)', f'Ridge (Best Alpha: {ridge_cv.alpha_:.3f})', f'Lasso (Best Alpha: {lasso_cv.alpha_:.3f})'],
    'MSE': [lr_mse, ridge_mse, lasso_mse],
    'R² Score': [lr_r2, ridge_r2, lasso_r2]
})

print("\n=== Model Performance Comparison ===")
print(results.to_string(index=False))

# ==========================================
# 4. Compare Coefficients (Shrinkage & Feature Selection)
# ==========================================
coef_df = pd.DataFrame({
    'Feature': diabetes.feature_names,
    'OLS Coef': lr_model.coef_,
    'Ridge Coef': ridge_cv.coef_,
    'Lasso Coef': lasso_cv.coef_
})

print("\n=== Model Coefficients Comparison ===")
print(coef_df.to_string(index=False))
