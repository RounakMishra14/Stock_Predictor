# Stock Prediction App - Stocket 🚀
 
**Stocket** is an interactive stock prediction app built using Streamlit, leveraging LSTM models and technical analysis tools to provide users with insights into stock price movements. The app allows users to visualize historical stock data, analyze technical indicators, and forecast future prices using a trained LSTM model.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Data Collection Process](#data-collection-process)
3. [LSTM Model Building](#lstm-model-building)
4. [App Features](#app-features)
5. [Technical Indicators](#technical-indicators)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [Future Enhancements](#future-enhancements)
10. [License](#license)
11. [Contact](#contact)

## 📘 Overview

Stocket uses historical stock data to build and deploy a predictive model that helps users visualize and predict stock price movements. It incorporates advanced data visualization and machine learning techniques, providing an easy-to-use interface for market analysis.

The project is built using:
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Modeling**: [Keras](https://keras.io/) LSTM Neural Network
- **Data Source**: [yfinance](https://pypi.org/project/yfinance/)
- **Visualization**: [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/)

## 📈 Data Collection Process

The data collection process utilizes the `yfinance` library, which pulls historical stock data for the selected stock ticker. To ensure the maximum available data is collected:
```python
# Set start date to a very early date
start = '1900-01-01'

# Set end date to today's date
end = datetime.datetime.today().strftime('%Y-%m-%d')
df = yf.download('SBIN.NS', start=start, end=end)
```

## 🧹 Data Preprocessing
Before building the LSTM model, the dataset undergoes several preprocessing steps to transform the raw data into a format suitable for training. The following steps are carried out:
```Python
data_trainning = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(data_trainning.shape, data_testing.shape)
```
The historical stock price data (df['Close']) is split into two parts:
* Training Set: 70% of the data is used for training the model (data_trainning).
* Testing Set: The remaining 30% is reserved for testing (data_testing).
  
This split ensures that the model is trained on the majority of the data while keeping a portion unseen for evaluation.
