import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Download historical stock price data
stock_symbol = "INCY"  # Change to your preferred stock symbol
df = yf.download(stock_symbol, start="2020-01-01", end="2024-03-17")

# Display first few rows``
print(df.head())




# Use only 'Close' prices for prediction
df['Prediction'] = df['Close'].shift(-30)  # Predict 30 days into the future

# Drop last 30 rows (as they won't have target values)
df.dropna(inplace=True)

# Features (X) and Target (y)
X = df[['Close']].values  # Using only Close price as feature
y = df['Prediction'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (important for ML models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual Prices", color='blue')
plt.plot(predictions, label="Predicted Prices", color='red', linestyle="dashed")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title(f"{stock_symbol} Stock Price Prediction")
plt.legend()
plt.show()


