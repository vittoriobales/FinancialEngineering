# -*- coding: utf-8 -*-
"""

@author: vitto
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Step 1: Data Preparation (Simplified with dummy data for illustration)
np.random.seed(42)  # For reproducible results
data = pd.DataFrame({
    'Day': range(365),
    'Price': np.sin(np.arange(365) * 0.1) + np.random.normal(0, 0.1, 365) + 10
})

# Assume 'Day' as a feature for simplicity and 'Price' as the target
X = data[['Day']]  # Features
y = data['Price']  # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Training the XGBoost Model
model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                         max_depth = 5, alpha = 10, n_estimators = 100)
model.fit(X_train, y_train)

# Forecasting future prices (Next 30 days for example)
X_future = pd.DataFrame({'Day': range(365, 395)})  # Dummy future days
y_future = model.predict(X_future)

# Step 3: Pricing an Exotic (Asian) Option using Monte Carlo simulation
n_simulations = 10000
T = 30 / 365  # Time to maturity in years, assuming 30 days
r = 0.05      # Risk-free rate
K = np.mean(y)  # Strike price, for illustration, use the mean of historical prices

# Simulating future price paths
sigma = np.std(y_future)  # Estimate volatility based on the future price predictions
daily_returns = np.exp((r - 0.5 * sigma**2) * T + sigma * np.random.normal(0, np.sqrt(T), (n_simulations, len(y_future))))

price_paths = np.zeros_like(daily_returns)
price_paths[:, 0] = y_future[0]  # Starting price for all simulations

for t in range(1, len(y_future)):
    price_paths[:, t] = price_paths[:, t-1] * daily_returns[:, t]

# Calculating the payoff for an Asian option
average_prices = np.mean(price_paths, axis=1)
payoffs = np.maximum(average_prices - K, 0)

# Discounting the payoffs to present value
option_price = np.exp(-r * T) * np.mean(payoffs)

print(f"MSE: {mean_squared_error(y_test, model.predict(X_test))}")
print(f"Exotic (Asian) Option Price: {option_price:.2f}")
