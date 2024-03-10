# -*- coding: utf-8 -*-
"""

@author: Vittorio Balestrieri
"""
import numpy as np
import matplotlib.pyplot as plt

# Model Parameters (adjusted for troubleshooting)
S0 = 50        # Initial spot price
mu = 55        # Long-term mean spot price
alpha = 0.1    # Speed of mean reversion
sigma_S = 0.3  # Increased volatility of the spot price for troubleshooting
T = 1          # Time to maturity
M = 100        # Number of time steps
dt = T / M     # Time step size
I = 1000       # Number of Monte Carlo paths
K = 40         # Adjusted strike price for the option for troubleshooting
r = 0.05       # Risk-free interest rate

# Correlation matrix and Cholesky decomposition for correlated Brownian motions
rho = -0.2  # Correlation between spot and yield
corr_matrix = np.array([[1, rho], [rho, 1]])
L = np.linalg.cholesky(corr_matrix)

np.random.seed(0)
Z = np.random.normal(size=(2, M, I))
Z_correlated = np.einsum('ij,jkl->ikl', L, Z)

def simulate_spot_prices(S0, mu, alpha, sigma_S, T, M, I, Z_correlated):
    S_paths = np.zeros((M + 1, I))
    S_paths[0] = S0
    for t in range(1, M + 1):
        Z_S = Z_correlated[0, t - 1]
        drift = alpha * (mu - S_paths[t - 1]) * dt
        shock = sigma_S * np.sqrt(dt) * Z_S
        S_paths[t] = S_paths[t - 1] + drift + shock
    return S_paths

S_paths = simulate_spot_prices(S0, mu, alpha, sigma_S, T, M, I, Z_correlated)

# Pricing an Asian Option
def price_Asian_option(S_paths, K, r, T):
    average_price = np.mean(S_paths, axis=0)
    payoffs = np.maximum(average_price - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

# Pricing an American Option using Least Squares Monte Carlo
def LSM_price(S_paths, K, r, dt):
    payoffs = np.maximum(K - S_paths, 0)
    V = np.zeros_like(payoffs)
    V[-1] = payoffs[-1]
    
    for t in reversed(range(M)):
        valid_indices = S_paths[t] > 0
        reg = np.polyfit(S_paths[t, valid_indices], V[t+1, valid_indices] * np.exp(-r*dt), deg=2)
        continuation_values = np.polyval(reg, S_paths[t, valid_indices])
        exercise_values = payoffs[t, valid_indices]
        V[t, valid_indices] = np.where(exercise_values > continuation_values, exercise_values, V[t+1, valid_indices] * np.exp(-r*dt))
    
    option_price = np.mean(V[1]) * np.exp(-r*dt)
    return option_price


asian_option_price = price_Asian_option(S_paths, K, r, T)
american_option_price = LSM_price(S_paths, K, r, dt)

# Print the range of average spot prices for troubleshooting
average_prices = np.mean(S_paths, axis=0)
print(f"Range of Average Spot Prices: Min = {np.min(average_prices):.2f}, Max = {np.max(average_prices):.2f}")

print(f"Asian Option Price: {asian_option_price:.2f}")
print(f"American Option Price: {american_option_price:.2f}")

# Plotting the first 5 simulated spot price paths
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(S_paths[:, i], label=f'Path {i+1}')
plt.title('Simulated Energy Spot Price Paths')
plt.xlabel('Time Step')
plt.ylabel('Spot Price')
plt.legend()
plt.show()
