import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. Load and Preprocess the Dataset
# ==========================================
print("Loading Fashion MNIST dataset...")
# Fetching from OpenML (Alternative: keras.datasets.fashion_mnist)
fashion = fetch_openml('Fashion-MNIST', version=1, parser='auto', as_frame=False)
X, y = fashion.data, fashion.target.astype(int)

# Normalize pixel values from [0, 255] to [0.0, 1.0]
X = X / 255.0

# Subsample data for efficient KNN training and evaluation 
# (KNN scales O(N) per test query, so we use 5,000 train / 1,000 test samples)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=5000, random_state=42, stratify=y)
X_test, _, y_test, _ = train_test_split(X_temp, y_temp, train_size=1000, random_state=42, stratify=y)

print(f"Training Set Shape: {X_train.shape}")
print(f"Testing Set Shape: {X_test.shape}")

# ==========================================
# 2. Experiment with Different Values of K
# ==========================================
k_values = [1, 3, 5, 7, 11, 15]
accuracies = []
models = {}

for k in k_values:
    print(f"Training KNN with K = {k}...")
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    knn.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    accuracies.append(acc)
    models[k] = (knn, y_pred)

# ==========================================
# 3. Performance Evaluation & Comparison
# ==========================================
results_df = pd.DataFrame({
    'K Value': k_values,
    'Accuracy': [f"{acc * 100:.2f}%" for acc in accuracies]
})

print("\n=== Accuracy Across Different K Values ===")
print(results_df.to_string(index=False))

# Detailed report for the optimal K
best_k = k_values[np.argmax(accuracies)]
print(f"\n--- Detailed Classification Report for Best K ({best_k}) ---")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(classification_report(y_test, models[best_k][1], target_names=class_names))

# ==========================================
# 4. Visualization of Accuracy vs. K
# ==========================================
plt.figure(figsize=(9, 5))
plt.plot(k_values, [acc * 100 for acc in accuracies], marker='o', color='dodgerblue', linewidth=2.5, markersize=8)
plt.title('Fashion MNIST: KNN Accuracy vs. Hyperparameter K', fontsize=14)
plt.xlabel('Value of K (Number of Neighbors)', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(k_values)
plt.show()
