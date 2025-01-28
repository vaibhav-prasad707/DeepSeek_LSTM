# ML-Based Stock Predictor for NVIDIA (NVDA)

## Overview
This project implements a machine learning-based stock predictor to analyze NVIDIA's (NVDA) stock performance and forecast its movement over the next 5 days. The model uses historical stock data, technical indicators, and sentiment analysis linked to the release of DeepSeek R1, a revolutionary AI chatbot. The goal is to evaluate the market impact of DeepSeek R1 and provide actionable insights using an LSTM-based time-series model.

---

## Features
1. **Data Collection & Preprocessing**
   - Fetches historical stock data for NVIDIA using `yfinance`.
   - Adds event-based binary features for the release of DeepSeek R1 (2025-01-20).
   - Incorporates sentiment analysis of news headlines related to DeepSeek R1.

2. **Feature Engineering**
   - Generates lagged features (e.g., 1–5 day lagged closing prices).
   - Computes technical indicators such as RSI (Relative Strength Index) and moving averages.
   - Uses sentiment scores and event flags to quantify news impact.

3. **Model Building**
   - Builds an LSTM-based time-series model to predict NVDA's stock price for the next 5 days.
   - Scales and sequences the data for effective time-series learning.

4. **Prediction & Visualization**
   - Forecasts NVDA's stock price movement for January 29, 2025, to February 2, 2025.
   - Visualizes historical stock performance alongside predicted values.

---

## Installation
### Prerequisites
- Python 3.8+
- Required Libraries:
  ```bash
  pip install yfinance pandas numpy matplotlib scikit-learn tensorflow
  ```

### Clone the Repository
```bash
git clone [<repository_url>](https://github.com/vaibhav-prasad707/DeepSeek_LSTM.git)
```

---

## Usage
### Step 1: Data Collection
Run the script to fetch NVIDIA's historical stock data and preprocess it. Ensure the event dates and sentiment data are correctly configured in the script.
```python
import yfinance as yf
import pandas as pd

# Fetch NVDA historical data
nvda = yf.Ticker("NVDA")
df = nvda.history(period="5y")
```

### Step 2: Feature Engineering
Generate lagged features, technical indicators, and sentiment scores:
```python
# Compute technical indicators and lagged features
df['MA_7'] = df['Close'].rolling(7).mean()
```

### Step 3: Model Building
Train the LSTM model using the processed data:
```python
# Build and train the LSTM model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
```

### Step 4: Prediction & Visualization
Use the trained model to forecast future stock prices and visualize the results:
```python
# Predict and plot results
plt.plot(df.index[-30:], df['Close'][-30:], label='Historical Close')
plt.plot(pred_dates, prediction, marker='o', linestyle='--', color='red', label='Predicted')
plt.show()
```

---


---

## Results
The model predicts a potential decline in NVIDIA's stock price following the release of DeepSeek R1. Visualizations provide a clear comparison between historical stock performance and forecasted trends.

---

## Future Enhancements
1. Improve sentiment analysis by integrating real-time scraping of financial news.
2. Expand the model to include other event-based features and economic indicators.
3. Optimize the LSTM architecture for better accuracy and generalizability.

---


## Author
Vaibhav Prasad  
Connect with me on [LinkedIn](www.linkedin.com/in/vaibhav-prasad-462b811aa)  
Check out my Medium article: [How DeepSeek R1 is Reshaping AI and Shaking Up the Stock Market](https://medium.com/@vaibhavprasad_52032/how-deepseek-r1-is-reshaping-ai-and-shaking-up-the-stock-market-88cf90a0baec)

---

## Acknowledgments
Special thanks to the open-source community for tools like `yfinance`, `VADER`, and TensorFlow, which made this project possible.
