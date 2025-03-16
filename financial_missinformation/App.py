import openai
import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import xgboost as xgb
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import date
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API Key (replace with your actual OpenAI API key)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI setup
st.title("Financial Misinformation Detector")
st.write("Enter financial news below, and the system will check its accuracy and provide supporting data.")

# Text input for the news article
input_news = st.text_area("Enter the financial news you want to check:")

# Function to query GPT-4 for real-time misinformation detection and supporting data
def query_gpt4(news_text):
    # GPT-4 API request for real-time accuracy check and supporting data
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial expert tasked with checking the accuracy of financial news and providing supporting data or sources."},
            {"role": "user", "content": f"Is this financial news accurate? Please provide supporting details and relevant context for: {news_text}, Answer only with either accurate or inaccurate"}
        ]
    )
    
    # Extract GPT-4's response
    gpt_output = response['choices'][0]['message']['content']
    # Analyze GPT-4's response to determine prediction (True/False) and extract evidence
    prediction = None
    if "a" in gpt_output[0] or "A" in gpt_output[0]:
        prediction = 1  # True news
    elif "i" in gpt_output[0] or "I" in gpt_output[0]:
        prediction = 0  # False news
    else:
        prediction = None  # Unable to determine
        
    return prediction, gpt_output

# Initialize a placeholder for predictions and evidence
predictions = []
supporting_evidence = []

# Check the news when the button is pressed
if st.button("Check Accuracy"):
    if input_news:
        # Query GPT-4 with the input news
        prediction, evidence = query_gpt4(input_news)
        
        # Display results based on GPT-4's output
        if prediction is not None:
            if prediction == 1:
                st.success("The news is likely accurate.")
                st.subheader("Supporting Evidence from GPT-4:")
                st.write(evidence)
            else:
                st.error("The news is likely false.")
                # Show supporting evidence from GPT-4
                st.subheader("Supporting Evidence from GPT-4:")
                st.write(evidence)
        else:
            st.warning("Unable to determine the accuracy of the news. Please check again or verify with other sources.")
    else:
        st.warning("Please enter some news text for evaluation.")

#Initial data requirement for Stock Market Prediction
st.title("Stock Market Predictor")
st.write("Enter a stock ticker symbol to predict its closing price.")

# User input
input_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, GOOGL):")

# Stock Prediction function
def predict_stock_price(ticker):
    # Fetch stock data
    stock = yf.Ticker(ticker)
    
    # Get first available date
    first_trade_epoch = stock.info.get("firstTradeDateEpochUtc", None)
    start_date = "2010-01-01"
    if first_trade_epoch:
        start_date = datetime.fromtimestamp(first_trade_epoch, tz=timezone.utc).strftime("%Y-%m-%d")
    end_date = date.today()

    # Load stock price data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        return None  # No data available
    
    # Use 'Close' price for prediction
    close_prices = stock_data["Close"].values.reshape(-1, 1)
    
    # Normalize prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(close_prices)

    # Create dataset
    def create_dataset(data, lag=5):
        X, y = [], []
        for i in range(len(data) - lag):
            X.append(data[i:i + lag, 0])
            y.append(data[i + lag, 0])
        return np.array(X), np.array(y)
    
    lag = 5
    X, y = create_dataset(scaled_close, lag)

    # Reshape for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Train-test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(lag, 1)),
        LSTM(units=50, return_sequences=False),
        Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Train the model (reduce epochs for speed)
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Predict using LSTM
    y_pred_lstm = model.predict(X_test)

    # Train XGBoost using actual prices (not LSTM output)
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50)
    xgb_model.fit(X_train.reshape(X_train.shape[0], lag), y_train)

    # Predict using XGBoost
    y_pred_xgb = xgb_model.predict(X_test.reshape(X_test.shape[0], lag))

    # Inverse transform to get actual stock prices
    predicted_price = scaler.inverse_transform([[y_pred_xgb[-1]]])[0][0]

    return predicted_price

# Prediction button
if st.button("Predict Stock Price"):
    if input_ticker:
        predicted_value = predict_stock_price(input_ticker.upper())
        
        if predicted_value:
            st.success(f"Predicted Closing Price: ${predicted_value:.2f}")
            st.warning("⚠️ This is an experimental prediction. Do your own research before investing.")
        else:
            st.error("Invalid stock ticker or no data available. Please try another stock.")
    else:
        st.warning("Please enter a valid stock ticker.")
