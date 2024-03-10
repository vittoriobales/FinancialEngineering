# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 11:36:31 2024

"""

import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Fetch commodity data from Yahoo Finance
def fetch_commodity_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Simulate correlated commodity price paths using Geometric Brownian Motion
def generate_correlated_commodity_paths(S0, mu, sigma, corr_matrix, T, dt, paths):
    n_commodities = len(S0)
    t = np.arange(0, T+dt, dt)
    paths_matrix = np.zeros((len(t), n_commodities, paths))
    paths_matrix[0] = np.array(S0).reshape(-1, 1)
    for i in range(1, len(t)):
        dW = np.random.multivariate_normal(mean=np.zeros(n_commodities), cov=corr_matrix*dt, size=paths).T
        paths_matrix[i] = paths_matrix[i-1] * np.exp((mu.reshape(-1, 1) - 0.5 * sigma.reshape(-1, 1)**2) * dt + sigma.reshape(-1, 1) * dW)
    return paths_matrix


# Calculate Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR)
def calculate_risk_measures(returns, confidence_level=0.95):
    var = np.percentile(returns, 100 * (1 - confidence_level), axis=0)
    cvar = np.mean(returns[returns <= var], axis=0)
    return var, cvar

# Portfolio optimization objective function
def portfolio_objective(weights, returns, lambda_reg=0.1):
    mean_return = np.dot(weights, np.mean(returns, axis=0))  # Calculate mean return without specifying axis
    returns_2d = np.reshape(returns, (returns.shape[1], -1))  # Reshape returns to 2D
    risk = np.dot(np.dot(weights, np.cov(returns_2d)), weights)
    penalty = lambda_reg * np.sqrt(risk)  # Regularization penalty
    return -((mean_return - penalty).sum())  # Return negative of the objective value as a scalar




# Portfolio optimization constraints
def portfolio_constraints(weights):
    return np.sum(weights) - 1

# Main function
def main():
    # Parameters
    np.random.seed(42)
    tickers = ['CL=F', 'GC=F', 'SI=F']  # Crude Oil, Gold, Silver futures
    start_date = '2020-01-01'
    end_date = '2021-01-01'
    S0 = fetch_commodity_data(tickers, start_date, end_date).iloc[0].values  # Initial prices
    mu = np.log(fetch_commodity_data(tickers, start_date, end_date).pct_change().mean() + 1).values  # Drifts
    sigma = fetch_commodity_data(tickers, start_date, end_date).pct_change().std().values  # Volatilities
    corr_matrix = fetch_commodity_data(tickers, start_date, end_date).pct_change().corr().values  # Correlation matrix
    T = 1  # Time horizon
    dt = 1/252  # Time step
    paths = 1000  # Number of paths
    confidence_level = 0.95  # Confidence level for risk measures
    lambda_reg = 0.1  # Regularization parameter

    # Generate correlated commodity price paths
    price_paths = generate_correlated_commodity_paths(S0, mu, sigma, corr_matrix, T, dt, paths)

    # Calculate returns
    returns = np.diff(price_paths, axis=0) / price_paths[:-1]

    # Calculate risk measures
    var, cvar = calculate_risk_measures(returns, confidence_level)
    print(f"Value-at-Risk (VaR) at {confidence_level*100}% confidence level:")
    print(var)
    print(f"Conditional Value-at-Risk (CVaR) at {confidence_level*100}% confidence level:")
    print(cvar)

    # Portfolio optimization
    initial_weights = np.ones(len(S0)) / len(S0)  # Equally weighted portfolio
    bounds = [(0, 1) for _ in range(len(S0))]  # Weight bounds
    constraints = {'type': 'eq', 'fun': portfolio_constraints}  # Sum of weights constraint
    optimal_weights = minimize(portfolio_objective, initial_weights, args=(returns, lambda_reg), bounds=bounds, constraints=constraints)

    print("\nOptimal portfolio weights:")
    print(optimal_weights.x)

    # Visualization
    for i in range(price_paths.shape[1]):  # Iterate over commodities
        plt.figure()
        plt.title(f"Commodity {i+1}")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        for j in range(10):  # Plot first 10 paths
            plt.plot(price_paths[:, i, j], label=f"Path {j+1}")
        plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

