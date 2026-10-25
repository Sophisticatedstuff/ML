import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. Load and Preprocess the Dataset
# ==========================================
# Load dataset directly from repository URL
url = "https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv"
df = pd.read_csv(url)

# Clean missing values if present in numeric columns
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna().reset_index(drop=True)

# Select Feature (displacement) and Target (mpg)
X = df[['displacement']].values  # Shape: (N, 1)
y = df['mpg'].values              # Shape: (N,)

# Split into Training (80%) and Testing (20%) sets for fair evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_split_size=0.2, random_state=42
)

# ==========================================
# 2. Polynomial Regression of Varying Degrees
# ==========================================
degrees = [1, 2, 3, 4]  # Degree 1 is standard Linear Regression
models = {}
results = []

for degree in degrees:
    # Transform feature to polynomial terms [1, x, x^2, ..., x^d]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Train Linear Regression model on polynomial features
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test_poly)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save model and transformer for plotting
    models[degree] = (model, poly)
    results.append({
        'Model': 'Linear Regression' if degree == 1 else f'Polynomial (Degree {degree})',
        'Degree': degree,
        'MSE': mse,
        'R2 Score': r2
    })

# ==========================================
# 3. Model Performance Comparison
# ==========================================
results_df = pd.DataFrame(results)
print("\n=== Model Evaluation Comparison (Test Set) ===")
print(results_df.to_string(index=False))

# ==========================================
# 4. Visualization of Fits
# ==========================================
# Create smooth line grid across the feature range
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

plt.figure(figsize=(10, 6))

# Plot raw scatter points
plt.scatter(X_train, y_train, color='gray', alpha=0.4, label='Train Data')
plt.scatter(X_test, y_test, color='black', alpha=0.7, label='Test Data', marker='x')

# Plot fitted curves for each degree
colors = ['red', 'blue', 'green', 'purple']
lines = ['--', '-', '-.', ':']

for idx, degree in enumerate(degrees):
    model, poly = models[degree]
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)
    
    label = f'Linear (Deg 1)' if degree == 1 else f'Poly Deg {degree}'
    plt.plot(X_range, y_range_pred, color=colors[idx], linestyle=lines[idx], 
             linewidth=2.5, label=f'{label} (R² = {results_df.loc[idx, "R2 Score"]:.3f})')

plt.title('Auto MPG: Engine Displacement vs. MPG (Linear vs Polynomial Fits)', fontsize=14)
plt.xlabel('Engine Displacement (cu. in.)', fontsize=12)
plt.ylabel('Miles Per Gallon (MPG)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
