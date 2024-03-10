import tkinter as tk
from tkinter import ttk
import yfinance as yf
import numpy as np
from scipy.stats import norm

asset_mapping = {
    "Gold": "GC=F",  # Example ticker symbol for gold futures
    "Oil": "CL=F",   # Example ticker symbol for crude oil futures
}

def retrieve_historical_data(underlying_asset_symbol, start_date, end_date):
    try:
        # Ensure underlying_asset_symbol is a string
        if not isinstance(underlying_asset_symbol, str):
            raise ValueError("Underlying asset symbol must be a string")

        data = yf.download(underlying_asset_symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        print("Error occurred while retrieving historical data for symbol '{}': {}".format(underlying_asset_symbol, e))
        return None



# Function to estimate volatility (simple implementation using historical volatility)
def estimate_volatility(data):
    # Placeholder implementation using simple historical volatility calculation
    volatility = data['Close'].pct_change().std() * (252 ** 0.5)  # Annualized volatility
    return volatility

# Function to calculate option Greeks (simple implementation)
def calculate_greeks(data, strike_price, risk_free_rate, volatility, time_to_maturity):
    # Calculate d1 and d2 for Black-Scholes formula
    d1 = (np.log(data['Close'].iloc[-1] / strike_price) + (risk_free_rate + (volatility ** 2) / 2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)

    # Calculate option Greeks
    delta = norm.cdf(d1) if option_type_var.get() == "Call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (data['Close'].iloc[-1] * volatility * np.sqrt(time_to_maturity))
    theta = -(data['Close'].iloc[-1] * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_maturity)) - risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2) if option_type_var.get() == "Call" else -(data['Close'].iloc[-1] * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_maturity)) + risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2)
    vega = data['Close'].iloc[-1] * np.sqrt(time_to_maturity) * norm.pdf(d1)
    rho = strike_price * time_to_maturity * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2) if option_type_var.get() == "Call" else -strike_price * time_to_maturity * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2)

    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}

# Function for risk analysis and management (simple implementation)
def analyze_risk():
    # Placeholder implementation for risk analysis and management
    return {'VaR': 1000, 'Delta Hedge': True, 'Stress Test Passed': True}

# Function for simulation and scenario analysis (simple implementation)
def perform_simulation():
    # Placeholder implementation for simulation and scenario analysis
    return {'Scenario 1': {'Option Price': 150, 'Delta': 0.6},
            'Scenario 2': {'Option Price': 120, 'Delta': 0.4}}
def calculate_option_price():
    # Get user inputs
    underlying_asset_name = underlying_asset_var.get()
    strike_price = float(strike_price_entry.get())
    expiration_years = float(expiration_date_entry.get())

    # Retrieve historical data
    underlying_asset_symbol = asset_mapping.get(underlying_asset_name)
    data = retrieve_historical_data(underlying_asset_symbol, start_date='2023-01-01', end_date='2024-01-01')
    if data is None:
        return

    # Estimate volatility
    volatility = estimate_volatility(data)

    # Calculate risk-free rate (placeholder value)
    risk_free_rate = 0.05  # Placeholder value, replace with actual risk-free rate

    # Calculate option price using Black-Scholes formula
    d1 = (np.log(data['Close'].iloc[-1] / strike_price) + (risk_free_rate + (volatility ** 2) / 2) * expiration_years) / (volatility * np.sqrt(expiration_years))
    d2 = d1 - volatility * np.sqrt(expiration_years)

    if option_type_var.get() == "Call":
        option_price = data['Close'].iloc[-1] * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * expiration_years) * norm.cdf(d2)
    elif option_type_var.get() == "Put":
        option_price = strike_price * np.exp(-risk_free_rate * expiration_years) * norm.cdf(-d2) - data['Close'].iloc[-1] * norm.cdf(-d1)
    else:
        option_price = 0

    # Calculate option Greeks
    option_greeks = calculate_greeks(data, strike_price, risk_free_rate, volatility, expiration_years)

    # Display the calculated option price
    option_price_label.config(text="Option Price: ${:.2f}".format(option_price))

    # Display the calculated Greeks
    greeks_label.config(text="Delta: {:.4f}, Gamma: {:.4f}, Theta: {:.4f}, Vega: {:.4f}, Rho: {:.4f}".format(
        option_greeks['Delta'], option_greeks['Gamma'], option_greeks['Theta'], option_greeks['Vega'], option_greeks['Rho']))

# Create main application window
root = tk.Tk()
root.title("Commodity Options Pricing Tool")

# Create option type dropdown
option_type_label = ttk.Label(root, text="Option Type:")
option_type_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")
option_type_var = tk.StringVar()
option_type_combobox = ttk.Combobox(root, textvariable=option_type_var, values=["Call", "Put"])
option_type_combobox.grid(row=0, column=1, padx=10, pady=5, sticky="w")
option_type_combobox.current(0)

# Create dropdown for selecting underlying asset
underlying_asset_label = ttk.Label(root, text="Underlying Asset:")
underlying_asset_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")
underlying_asset_var = tk.StringVar()
underlying_asset_combobox = ttk.Combobox(root, textvariable=underlying_asset_var, values=["Gold", "Oil"])
underlying_asset_combobox.grid(row=1, column=1, padx=10, pady=5, sticky="w")
underlying_asset_combobox.current(0)

# Create input fields for strike price and expiration date
strike_price_label = ttk.Label(root, text="Strike Price:")
strike_price_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")
strike_price_entry = ttk.Entry(root)
strike_price_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")

expiration_date_label = ttk.Label(root, text="Expiration Date (years):")
expiration_date_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")
expiration_date_entry = ttk.Entry(root)
expiration_date_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")


# Create button to calculate option price
calculate_button = ttk.Button(root, text="Calculate Option Price", command=calculate_option_price)
calculate_button.grid(row=4, columnspan=2, padx=10, pady=10)

# Label to display option price
option_price_label = ttk.Label(root, text="")
option_price_label.grid(row=5, columnspan=2, padx=10, pady=5)

greeks_label = ttk.Label(root, text="")
greeks_label.grid(row=6, columnspan=2, padx=10, pady=5)
# Run the application
root.mainloop()

