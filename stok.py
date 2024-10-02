# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Fetch stock data (For example, 'AAPL' - Apple Inc.)
stock_symbol = 'AAPL'
stock_data = yf.download(stock_symbol, start="2015-01-01", end="2023-01-01")

# Feature Engineering: Using Adjusted Close Price as Target Variable
stock_data['Prediction'] = stock_data['Adj Close'].shift(-30)  # Predicting 30 days ahead

# Prepare the dataset
X = np.array(stock_data[['Adj Close']])[:-30]
y = np.array(stock_data['Prediction'])[:-30]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build the model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Adj Close'][-len(y_test):], label='Actual Prices')
plt.plot(range(len(stock_data) - len(y_test), len(stock_data)), predictions, label='Predicted Prices', linestyle='--')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
