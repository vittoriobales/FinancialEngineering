# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 11:06:48 2024

@author: vitto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Data Collection
def fetch_oil_price(symbol, start_date, end_date):
    """
    Fetch historical oil price data from Yahoo Finance.
    """
    oil_data = yf.download(symbol, start=start_date, end=end_date)
    return oil_data

# Step 2: Data Preprocessing
def preprocess_data(data):
    """
    Preprocess the fetched data.
    """
    try:
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        data_copy = data.copy()

        # Replace infinite values with NaNs
        data_copy.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Replace NaN values with the mean of the column
        data_copy.fillna(data_copy.mean(), inplace=True)

        # Calculate returns
        data_copy['Return'] = data_copy['Adj Close'].pct_change()

        # Drop NaN values in the 'Return' column
        data_copy.dropna(subset=['Return'], inplace=True)

        # Print first few rows of the DataFrame for inspection
        print("Preprocessed Data:")
        print(data_copy.head())

        return data_copy
    except Exception as e:
        print("Error during preprocessing:", e)


# Step 3: Machine Learning - Predict Optimal Stopping Point
def predict_optimal_stopping_point(data):
    """
    Predict the optimal stopping point using a machine learning model.
    """
    try:
        # Feature engineering
        data['Return'] = data['Adj Close'].pct_change()
        data.dropna(inplace=True)

        # Define features and target variable
        X = data[['Adj Close', 'Return']].values
        y = data['Adj Close'].shift(-1).values

        # Check for NaN values in target variable and remove corresponding rows from features and target
        nan_indices = np.isnan(y)
        X = X[~nan_indices]
        y = y[~nan_indices]

        # Print shape of features and target variable after removing NaN values
        print("Shape of Features (X):", X.shape)
        print("Shape of Target Variable (y):", y.shape)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a random forest regressor
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_train)

        # Predict optimal stopping point
        predicted_prices = rf_regressor.predict(X_test)
        mse = mean_squared_error(y_test, predicted_prices)
        
        print("Predicted Prices:")
        print(predicted_prices)
        
        return predicted_prices, mse
    except Exception as e:
        print("Error during prediction:", e)

# Step 4: Simulation and Optimal Stopping Strategy
def simulate_trading(data, predicted_prices):
    """
    Simulate trading based on optimal stopping strategy.
    """
    try:
        # Assume simple strategy: buy when current price is below predicted price, sell otherwise
        data['Predicted Price'] = np.nan
        data.loc[data.index[-len(predicted_prices):], 'Predicted Price'] = predicted_prices

        data['Action'] = np.where(data['Adj Close'] < data['Predicted Price'], 'Buy', 'Sell')

        # Calculate returns
        data['Return'] = data['Adj Close'].pct_change()
        data['Strategy Return'] = data['Return'] * data['Action'].apply(lambda x: 1 if x == 'Buy' else -1)

        # Calculate cumulative returns
        data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()
        
        print("Simulated Data:")
        print(data.head())
        
        return data
    except Exception as e:
        print("Error during simulation:", e)
        
def calculate_max_drawdown(data):
    """
    Calculate the maximum drawdown.
    """
    cum_returns = data['Cumulative Strategy Return']
    max_return = cum_returns.cummax()
    drawdown = (cum_returns - max_return) / max_return
    max_drawdown = drawdown.min()
    return max_drawdown


# Main function
def main():
    try:
        # Define parameters
        symbol = 'CL=F'  # Oil futures symbol on Yahoo Finance
        start_date = '2023-01-01'
        end_date = '2023-12-31'

        # Fetch data
        oil_data = fetch_oil_price(symbol, start_date, end_date)

        # Preprocess data
        preprocessed_data = preprocess_data(oil_data)

        # Inspect preprocessed data
        print("Preprocessed Data Shape:")
        print(preprocessed_data.shape)
        print("NaN Values in Preprocessed Data:")
        print(preprocessed_data.isnull().sum())

        # Machine Learning - Predict Optimal Stopping Point
        predicted_prices, mse = predict_optimal_stopping_point(preprocessed_data)

        # Simulation and Optimal Stopping Strategy
        simulated_data = simulate_trading(preprocessed_data, predicted_prices)

        # Risk Management
        max_drawdown = calculate_max_drawdown(simulated_data)

        # Visualize data
        plt.figure(figsize=(10, 6))
        plt.plot(simulated_data.index, simulated_data['Adj Close'], label='Oil Price')
        plt.plot(simulated_data.index, simulated_data['Predicted Price'], label='Predicted Price')
        plt.title('Historical Oil Price and Predicted Optimal Stopping Point')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print risk metrics
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Maximum Drawdown: {max_drawdown}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
