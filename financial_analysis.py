import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

# Step 1: Financial News Scraping
def get_financial_news(url="https://finance.yahoo.com/"):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('h3', class_='Mb(5px)')
    news_data = []
    for article in articles:
        title = article.text
        link = article.find('a')['href']
        news_data.append({"Title": title, "Link": f"https://finance.yahoo.com{link}"})
    return pd.DataFrame(news_data)

# Step 2: Sentiment Analysis
def analyze_sentiments(news_df):
    finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    news_df['Sentiment'] = news_df['Title'].apply(lambda x: finbert(x)[0]['label'])
    return news_df

# Step 3: Stock Data Collection
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Date'] = stock_data.index
    return stock_data

# Step 4: Merge Data
def merge_data(sentiment_data, stock_data):
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'], errors='coerce').dt.date
    stock_data['Date'] = stock_data['Date'].dt.date
    return pd.merge(stock_data, sentiment_data, on='Date', how='inner')

# Step 5: Prepare Data for Time-Series Prediction
def prepare_data(data, feature_col, target_col, look_back=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[[feature_col]].values)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(data[target_col].values[i])
    return np.array(X), np.array(y), scaler

# Step 6: LSTM Model for Stock Prediction
def build_and_train_model(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    return model

# Step 7: Visualize Results
def visualize_predictions(actual, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='red')
    plt.legend()
    plt.title("Stock Price Prediction")
    plt.show()

# Step 8: Flask API for Sentiment Analysis
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    sentiment = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")(text)[0]
    return jsonify(sentiment)

# Main Function
if __name__ == '__main__':
    # Step 1: Scrape Financial News
    news_df = get_financial_news()
    print("Collected Financial News:")
    print(news_df.head())
    
    # Step 2: Analyze Sentiments
    sentiment_df = analyze_sentiments(news_df)
    print("Sentiment Analysis Completed:")
    print(sentiment_df.head())
    
    # Step 3: Fetch Stock Data
    stock_df = get_stock_data("AAPL", "2023-01-01", "2023-12-31")
    print("Stock Data Retrieved:")
    print(stock_df.head())
    
    # Step 4: Merge Sentiment and Stock Data
    merged_df = merge_data(sentiment_df, stock_df)
    print("Merged Data:")
    print(merged_df.head())
    
    # Step 5: Prepare Data for Prediction
    look_back = 5
    X, y, scaler = prepare_data(stock_df, 'Close', 'Close', look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM

    # Step 6: Train Model
    lstm_model = build_and_train_model(X, y)

    # Step 7: Predict and Visualize
    predicted_prices = lstm_model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    visualize_predictions(stock_df['Close'][look_back:], predicted_prices)

    # Step 8: Start Flask API
    app.run(debug=True)
