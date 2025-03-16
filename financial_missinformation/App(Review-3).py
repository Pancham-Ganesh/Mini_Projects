import openai
import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import date
from datetime import datetime, timezone

# Set OpenAI API Key (replace with your actual OpenAI API key)
openai.api_key = 'openAI API key'

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
st.write("Provide the supporting data.")
input_ticker = st.text_area("Enter the Stock market name to predict it's value: ")

#Stock Prediction function
def stocks(stockName):
    ticker = stockName
    stock = yf.Ticker(ticker)
    first_trade_epoch = stock.info.get("firstTradeDateEpoch", None)
    start_date = "2010-01-01"
    if first_trade_epoch:
        start_date = datetime.fromtimestamp(first_trade_epoch, tz=timezone.utc).strftime("%Y-%m-%d")
    end_date = date.today()
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(stock_data["Close"].values.reshape(-1, 1))

    def create_dataset(data, lag=1):
        X, y = [], []
        for i in range(len(data) - lag - 1):
            X.append(data[i:(i + lag), 0])
            y.append(data[i + lag, 0])
        return np.array(X), np.array(y)
    
    lag = 5
    X, y = create_dataset(scaled_close, lag)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

    y_pred_lstm = model.predict(X_test)
    y_pred_lstm_2d = y_pred_lstm[:, -1, :]

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(y_pred_lstm_2d, y_test)

    y_pred_boosted = xgb_model.predict(y_pred_lstm_2d)

    y_pred_boosted_actual = scaler.inverse_transform(y_pred_boosted.reshape(-1, 1))

    predicted_closing_price = y_pred_boosted_actual[-1][0]
    return predicted_closing_price

predictedValue = 0

# Check the news when the button is pressed
if st.button("Check Stock Market Prediction"):
    if input_ticker:
        # Query for Stock Market Prediction
        predictedValue = stocks(input_ticker)

        if predictedValue!=0:
            st.write("The stock will close at the following predicted value: ")
            st.success(predictedValue)
            st.warning("The product is in testing phase, please do your research for accurate results and no losses.")

    else:
        st.warning("Please enter the correct Stock Name.")        
