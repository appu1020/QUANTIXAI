import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Fetch historical stock data
def get_stock_data(stock_symbol, start_date="2015-01-01"):
    stock = yf.download(stock_symbol, start=start_date)
    return stock[['Close']]  # Only keep closing prices

# Get data for a specific stock (e.g., "RELIANCE.NS" for Reliance Industries)
stock_symbol = "RELIANCE.NS"  # NSE India format
df = get_stock_data(stock_symbol)
df.head()



scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Convert data into sequences for LSTM
def create_sequences(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60  # Look back 60 days
X, Y = create_sequences(df_scaled, time_step)

# Reshape for LSTM (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into training (80%) and testing (20%)
split = int(0.8 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]


model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)


import plotly.graph_objects as go
import numpy as np

# Downsample data by selecting every n-th point
n = 5  # Adjust to control density (higher n = fewer points)
x_values = df.index[-len(Y_test):][::n]
actual_y_values = actual_prices.flatten()[::n]
predicted_y_values = predictions.flatten()[::n]

# Create interactive figure
fig = go.Figure()

# Add actual prices trace
fig.add_trace(go.Scatter(
    x=x_values, 
    y=actual_y_values, 
    mode='lines+markers', 
    name="Actual Price",
    hoverinfo="x+y+text",
    text=[f"Actual: {y:.2f}" for y in actual_y_values],  # Data labels in hover tooltip
    marker=dict(size=7)  # Adjust marker size
))

# Add predicted prices trace
fig.add_trace(go.Scatter(
    x=x_values, 
    y=predicted_y_values, 
    mode='lines+markers', 
    name="Predicted Price",
    hoverinfo="x+y+text",
    text=[f"Predicted: {y:.2f}" for y in predicted_y_values],  # Data labels in hover tooltip
    marker=dict(size=7, symbol="square")  # Different marker for distinction
))

# Customize layout
fig.update_layout(
    title=f"Stock Price Prediction for {stock_symbol}",
    xaxis_title="Date",
    yaxis_title="Stock Price",
    hovermode="x unified",  # Unifies hover info across traces
    template="plotly_white"  # Clean white background
)

fig.show()


# Get last 60 days' data
last_60_days = df_scaled[-time_step:]
last_60_days = last_60_days.reshape(1, time_step, 1)

# Predict next day price
predicted_price = model.predict(last_60_days)
predicted_price = scaler.inverse_transform(predicted_price)[0][0]

print(f"Predicted next day's price for {stock_symbol}: ₹{predicted_price:.2f}")


# Compute Relative Strength Index (RSI)
def compute_RSI(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

df['RSI'] = compute_RSI(df)

# Compute Moving Averages (50-day & 200-day)
df['MA_50'] = df['Close'].rolling(window=50).mean()
df['MA_200'] = df['Close'].rolling(window=200).mean()

# Market Sentiment Analysis using VADER (Example with News Headlines)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

def fetch_news_sentiment(stock_symbol):
    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&apiKey=YOUR_NEWSAPI_KEY"
    response = requests.get(url).json()
    analyzer = SentimentIntensityAnalyzer()
    
    sentiment_scores = [analyzer.polarity_scores(article['title'])['compound'] for article in response.get('articles', [])]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return avg_sentiment

market_sentiment = fetch_news_sentiment(stock_symbol)
print(f"Market Sentiment for {stock_symbol}: {market_sentiment}")

