import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Load and Preprocess the Dataset
# ==========================================
print("Loading 20 Newsgroups dataset subset...")
# Select a few distinct categories for clear demonstration
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
dataset = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'), random_state=42)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.25, random_state=42
)

# Vectorize text into term counts (Bag of Words)
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_train_counts = vectorizer.fit_transform(X_train_text).toarray()
X_test_counts = vectorizer.transform(X_test_text).toarray()

vocab_size = X_train_counts.shape[1]
print(f"Vocabulary Size (V): {vocab_size}")
print(f"Training Documents: {X_train_counts.shape[0]}")

# ==========================================
# 2. Implement MLE and MAP Estimators
# ==========================================
def estimate_mle(counts):
    """Calculates Maximum Likelihood Estimates for multinomial parameters."""
    total_counts = np.sum(counts)
    # Add a tiny epsilon or handle zeros for safety, but raw MLE keeps zeros
    return counts / total_counts

def estimate_map(counts, k=1.0):
    """Calculates MAP Estimates with symmetric Dirichlet prior (Additive smoothing k)."""
    total_counts = np.sum(counts)
    # k represents pseudocounts (alpha - 1)
    return (counts + k) / (total_counts + vocab_size * k)

# Aggregate global word counts across all training documents
global_counts = np.sum(X_train_counts, axis=0)

# Compute parameters under different estimators
theta_mle = estimate_mle(global_counts)
theta_map_laplace = estimate_map(global_counts, k=1.0)  # Laplace prior (k=1)
theta_map_sparse = estimate_map(global_counts, k=0.1)   # Sparse prior (k=0.1)
theta_map_smooth = estimate_map(global_counts, k=10.0)  # Strong smoothing prior (k=10)

# ==========================================
# 3. Evaluate Impact on Zero-Counts & Perplexity
# ==========================================
def evaluate_log_likelihood(theta, test_counts):
    """Calculates average log-likelihood of test data given parameter theta."""
    # Avoid log(0) for MLE by clipping or handling zero probabilities
    eps = 1e-12
    safe_theta = np.clip(theta, eps, 1.0)
    
    log_likelihoods = []
    for doc in test_counts:
        if np.sum(doc) > 0:
            # Multinomial log-likelihood per document
            ll = np.sum(doc * np.log(safe_theta))
            log_likelihoods.append(ll / np.sum(doc))
            
    return np.mean(log_likelihoods)

ll_mle = evaluate_log_likelihood(theta_mle, X_test_counts)
ll_laplace = evaluate_log_likelihood(theta_map_laplace, X_test_counts)
ll_sparse = evaluate_log_likelihood(theta_map_sparse, X_test_counts)
ll_smooth = evaluate_log_likelihood(theta_map_smooth, X_test_counts)

results = pd.DataFrame({
    'Estimator / Prior': [
        'MLE (No Prior / k=0)', 
        'MAP - Laplace (Uniform Prior / k=1)', 
        'MAP - Sparse Prior (k=0.1)', 
        'MAP - Heavy Smoothing (k=10)'
    ],
    'Zero Probabilities Handled?': ['No (Assigns 0)', 'Yes', 'Yes', 'Yes'],
    'Avg Test Log-Likelihood': [
        f"{ll_mle:.4f} (Undefined if OOV)", 
        f"{ll_laplace:.4f}", 
        f"{ll_sparse:.4f}", 
        f"{ll_smooth:.4f}"
    ]
})

print("\n=== Model Evaluation & Comparison ===")
print(results.to_string(index=False))

# ==========================================
# 4. Inspecting Parameter Shifts (Top Words)
# ==========================================
feature_names = vectorizer.get_feature_names_out()
top_indices = np.argsort(global_counts)[::-1][:10]

comparison_df = pd.DataFrame({
    'Word': [feature_names[i] for i in top_indices],
    'Raw Count': global_counts[top_indices],
    'MLE Prob': theta_mle[top_indices],
    'MAP (Laplace) Prob': theta_map_laplace[top_indices],
    'MAP (Heavy) Prob': theta_map_smooth[top_indices]
})

print("\n=== Top 10 Frequent Words & Parameter Estimates ===")
print(comparison_df.to_string(index=False))
