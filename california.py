import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

def main():
    # ==========================================
    # 1. Load and Preprocess the Dataset
    # ==========================================
    print("Loading California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # Filter out extreme outliers for clean visualization & training stability
    df = df[df['AveRooms'] < 10].copy()

    X_raw = df[['AveRooms']].values  # Shape: (N, 1)
    y = df['MedHouseVal'].values.reshape(-1, 1)  # Shape: (N, 1)
    N = len(y)

    # Standardize Feature (Crucial for Gradient Descent convergence)
    X_mean = np.mean(X_raw)
    X_std = np.std(X_raw)
    X_scaled = (X_raw - X_mean) / X_std

    # Add Intercept Column (Bias term X0 = 1)
    X_b = np.c_[np.ones((N, 1)), X_scaled]  # Shape: (N, 2)

    # ==========================================
    # 2. Implementations
    # ==========================================
    
    # Method 1: Closed-Form Normal Equation
    def normal_equation(X, y):
        # theta = (X^T * X)^(-1) * X^T * y
        return np.linalg.inv(X.T @ X) @ X.T @ y

    theta_normal = normal_equation(X_b, y)

    # Method 2: Batch Gradient Descent
    def gradient_descent(X, y, alpha=0.05, iterations=1000):
        N = len(y)
        theta = np.zeros((2, 1))  # Initialize weights [theta_0, theta_1] to zero
        cost_history = []

        for _ in range(iterations):
            gradients = (1 / N) * X.T @ (X @ theta - y)
            theta -= alpha * gradients
            
            # Calculate Cost J(theta) - MSE
            cost = (1 / (2 * N)) * np.sum((X @ theta - y) ** 2)
            cost_history.append(cost)

        return theta, cost_history

    theta_gd, cost_history = gradient_descent(X_b, y, alpha=0.05, iterations=1000)

    # ==========================================
    # 3. Model Evaluation
    # ==========================================
    def evaluate_model(X, y, theta):
        y_pred = X @ theta
        mse = np.mean((y - y_pred) ** 2)
        
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return mse, r2

    mse_norm, r2_norm = evaluate_model(X_b, y, theta_normal)
    mse_gd, r2_gd = evaluate_model(X_b, y, theta_gd)

    print("\n--- Normal Equation Results ---")
    print(f"Theta: Bias = {theta_normal[0][0]:.4f}, Weight = {theta_normal[1][0]:.4f}")
    print(f"MSE: {mse_norm:.4f} | R² Score: {r2_norm:.4f}")

    print("\n--- Gradient Descent Results ---")
    print(f"Theta: Bias = {theta_gd[0][0]:.4f}, Weight = {theta_gd[1][0]:.4f}")
    print(f"MSE: {mse_gd:.4f} | R² Score: {r2_gd:.4f}")

    # ==========================================
    # 4. Visualization
    # ==========================================
    x_plot_raw = np.linspace(X_raw.min(), X_raw.max(), 100).reshape(-1, 1)
    x_plot_scaled = (x_plot_raw - X_mean) / X_std
    x_plot_b = np.c_[np.ones((100, 1)), x_plot_scaled]
    y_plot_pred = x_plot_b @ theta_gd

    plt.figure(figsize=(10, 6))
    plt.scatter(X_raw, y, alpha=0.2, color='dodgerblue', label='California Housing Blocks')
    plt.plot(x_plot_raw, y_plot_pred, color='red', linewidth=2.5, label='Linear Regression Fit (GD)')

    plt.title('California Housing: Average Rooms vs. Median House Value', fontsize=14)
    plt.xlabel('Average Number of Rooms per Dwelling (AveRooms)', fontsize=12)
    plt.ylabel('Median House Value ($100,000s)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main()
