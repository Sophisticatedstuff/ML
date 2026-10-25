import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ==========================================
# 1. Load and Preprocess the Dataset
# ==========================================
print("Loading 20 Newsgroups subset...")
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']

# Load train and test splits, stripping email headers, footers, and quoted replies
train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'), random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'), random_state=42)

X_train_text, y_train = train_data.data, train_data.target
X_test_text, y_test = test_data.data, test_data.target

# --- Representation A: Count Vectors for Multinomial NB ---
vectorizer_count = CountVectorizer(max_features=5000, stop_words='english')
X_train_counts = vectorizer_count.fit_transform(X_train_text)
X_test_counts = vectorizer_count.transform(X_test_text)

# --- Representation B: Binary Vectors for Bernoulli NB ---
vectorizer_binary = CountVectorizer(max_features=5000, stop_words='english', binary=True)
X_train_binary = vectorizer_binary.fit_transform(X_train_text)
X_test_binary = vectorizer_binary.transform(X_test_text)

# ==========================================
# 2. Implement & Train Naïve Bayes Models
# ==========================================
# A. Multinomial Naïve Bayes
mnb_model = MultinomialNB(alpha=1.0)
mnb_model.fit(X_train_counts, y_train)
mnb_preds = mnb_model.predict(X_test_counts)

# B. Bernoulli Naïve Bayes
bnb_model = BernoulliNB(alpha=1.0)
bnb_model.fit(X_train_binary, y_train)
bnb_preds = bnb_model.predict(X_test_binary)

# ==========================================
# 3. Model Evaluation & Performance Comparison
# ==========================================
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    return {
        'Model': model_name,
        'Accuracy': f"{acc * 100:.2f}%",
        'Macro F1-Score': f"{f1_macro:.4f}",
        'Weighted F1-Score': f"{f1_weighted:.4f}"
    }

results = [
    evaluate_model(y_test, mnb_preds, 'Multinomial Naïve Bayes'),
    evaluate_model(y_test, bnb_preds, 'Bernoulli Naïve Bayes')
]

results_df = pd.DataFrame(results)
print("\n=== Performance Metrics Comparison ===")
print(results_df.to_string(index=False))

print("\n--- Multinomial Naïve Bayes Detailed Report ---")
print(classification_report(y_test, mnb_preds, target_names=train_data.target_names))

print("--- Bernoulli Naïve Bayes Detailed Report ---")
print(classification_report(y_test, bnb_preds, target_names=train_data.target_names))
