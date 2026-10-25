import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. Load and Preprocess the Online Retail Dataset
# ==========================================
print("Loading Online Retail dataset...")
url = "https://raw.githubusercontent.com/plotly/datasets/master/online_retail.csv"
df = pd.read_csv(url)

# Clean dataset: Drop missing CustomerIDs and cancelled invoices (InvoiceNo starting with 'C')
df = df.dropna(subset=['CustomerID'])
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Calculate Total Price per transaction
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Aggregate into RFM (Recency, Frequency, Monetary) + Average Basket Size
rfm_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'InvoiceNo': 'nunique',                                  # Frequency
    'TotalPrice': ['sum', 'mean']                            # Monetary & Avg Spend
})

rfm_df.columns = ['Recency', 'Frequency', 'Monetary', 'AvgBasketValue']
rfm_df = rfm_df.reset_index()

# Create Target: High-Value Customer (1 if Monetary spend > Median, else 0)
median_spend = rfm_df['Monetary'].median()
rfm_df['HighValueCustomer'] = (rfm_df['Monetary'] > median_spend).astype(int)

# Features and Target
X = rfm_df[['Recency', 'Frequency', 'Monetary', 'AvgBasketValue']]
y = rfm_df['HighValueCustomer']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 2. Implement ID3 Decision Tree Classifier
# ==========================================
# Note: criterion='entropy' uses Information Gain (the core mathematical objective of ID3)
id3_tree = DecisionTreeClassifier(
    criterion='entropy', 
    max_depth=4, 
    random_state=42
)

id3_tree.fit(X_train, y_train)
y_pred = id3_tree.predict(X_test)

# ==========================================
# 3. Model Evaluation
# ==========================================
acc = accuracy_score(y_test, y_pred)
print(f"\n=== Decision Tree (ID3 / Entropy) Evaluation ===")
print(f"Test Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# ==========================================
# 4. Feature Importance Analysis
# ==========================================
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Information Gain / Importance': id3_tree.feature_importances_
}).sort_values(by='Information Gain / Importance', ascending=False)

print("\n=== Feature Importances ===")
print(importance_df.to_string(index=False))

# ==========================================
# 5. Visualize the Decision Tree
# ==========================================
plt.figure(figsize=(14, 8))
plot_tree(
    id3_tree, 
    feature_names=X.columns.tolist(), 
    class_names=['Low Value', 'High Value'], 
    filled=True, 
    rounded=True,
    fontsize=10
)
plt.title('ID3 Decision Tree Structure for Customer Segmentation', fontsize=14)
plt.show()
