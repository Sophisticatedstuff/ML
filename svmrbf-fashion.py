import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. Load and Preprocess the Dataset
# ==========================================
print("Loading Fashion MNIST dataset...")
fashion = fetch_openml('Fashion-MNIST', version=1, parser='auto', as_frame=False)
X, y = fashion.data, fashion.target.astype(int)

# Normalize pixel values from [0, 255] to [0.0, 1.0]
X = X / 255.0

# Subsample for efficient SVM training (5,000 train / 1,000 test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=5000, random_state=42, stratify=y)
X_test, _, y_test, _ = train_test_split(X_temp, y_temp, train_size=1000, random_state=42, stratify=y)

print(f"Training Set Shape: {X_train.shape}")
print(f"Testing Set Shape: {X_test.shape}")

# ==========================================
# 2. Implement SVM with Different Kernels
# ==========================================
kernels = {
    'Linear': SVC(kernel='linear', C=1.0, random_state=42),
    'Polynomial (Degree 3)': SVC(kernel='poly', degree=3, C=1.0, gamma='scale', random_state=42),
    'RBF (Gaussian)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
}

results = []
models = {}

for name, model in kernels.items():
    print(f"Training SVM with {name} kernel...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    models[name] = y_pred
    results.append({
        'Kernel': name,
        'Test Accuracy': f"{acc * 100:.2f}%"
    })

# ==========================================
# 3. Performance Comparison
# ==========================================
results_df = pd.DataFrame(results)
print("\n=== SVM Kernel Performance Comparison ===")
print(results_df.to_string(index=False))

# Detailed report for RBF Kernel
print("\n--- Detailed Classification Report (RBF Kernel) ---")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(classification_report(y_test, models['RBF (Gaussian)'], target_names=class_names))
