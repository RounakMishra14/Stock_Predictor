import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import plotly.graph_objs as go

import datetime

# Set start date to a very early date
start = '1900-01-01'

# Set end date to today's date
end = datetime.datetime.today().strftime('%Y-%m-%d')


st.title('Stock Analyzer')

user_input= st.text_input('Enter Stock Ticker', 'SBIN.NS')
df=yf.download(user_input, start=start, end=end)

st.subheader('Summary Stats of the Stock:')
st.write(df.describe())

st.subheader('Closing Price Vs Time')
fig= plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Trading Volume Over Time')
fig_volume = plt.figure(figsize=(12, 6))
plt.plot(df['Volume'], label='Volume', color='orange')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Trading Volume Over Time')
plt.legend()
st.pyplot(fig_volume)


st.subheader('MA100 with MA200')
fig= plt.figure(figsize=(12, 6))
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
plt.plot(df.Close)
plt.plot(ma100, color='r')
plt.plot(ma200, color='g')
st.pyplot(fig)


st.subheader('Candlestick Chart')

# Plot the candlestick chart using Plotly
fig_candlestick = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])

fig_candlestick.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig_candlestick, key='original_candlestick')

# Add a slider for the date range below the candlestick chart
min_date = df.index.min()
max_date = df.index.max()

# Slider for selecting date range
date_range = st.slider(
    "Select date range for zoom",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM-DD"
)

# Filter the DataFrame based on the selected date range
filtered_df = df.loc[(df.index >= date_range[0]) & (df.index <= date_range[1])]

# Plot the filtered candlestick chart
fig_zoomed_candlestick = go.Figure(data=[go.Candlestick(
    x=filtered_df.index,
    open=filtered_df['Open'],
    high=filtered_df['High'],
    low=filtered_df['Low'],
    close=filtered_df['Close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])

fig_zoomed_candlestick.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig_zoomed_candlestick, key='zoomed_candlestick')



def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df)
st.subheader('RSI (Relative Strength Index)')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(df['RSI'], label='RSI')
plt.axhline(70, linestyle='--', color='r')
plt.axhline(30, linestyle='--', color='g')
plt.legend()
st.pyplot(fig4)



def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

df['MACD'], df['Signal'] = calculate_macd(df)
st.subheader('MACD (Moving Average Convergence Divergence)')
fig5 = plt.figure(figsize=(12, 6))
plt.plot(df['MACD'], label='MACD', color='b')
plt.plot(df['Signal'], label='Signal', color='r')
plt.legend()
st.pyplot(fig5)



data_trainning=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler=MinMaxScaler(feature_range=(0, 1))

data_trainning_array=scaler.fit_transform(data_trainning)



model= load_model('LSTM_model.h5')

past_100_days = data_trainning.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index= True)

input_data= scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test= np.array(x_test), np.array(y_test)

y_predicted=model.predict(x_test)

scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

st.subheader('Predictions Vs Original')
fig2=plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original_Price')
plt.plot(y_predicted, 'r', label='Predicted_Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

def forecast_next_days(model, data, scaler, days=15):
    last_100_days = data[-100:].reshape(-1, 1)
    new_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = new_scaler.fit_transform(last_100_days)
    x_input = np.array(scaled_data).reshape(1, -1, 1)
    forecasted_prices = []
    for _ in range(days):
        next_price = model.predict(x_input)[0][0]
        forecasted_prices.append(next_price)
        scaled_data = np.append(scaled_data, next_price)
        scaled_data = scaled_data[-100:]
        x_input = scaled_data.reshape(1, -1, 1)
    forecasted_prices = np.array(forecasted_prices).reshape(-1, 1)
    forecasted_prices = new_scaler.inverse_transform(forecasted_prices)
    return forecasted_prices.flatten()

# Get the last date in the dataset and generate the next 15 dates
last_date = df.index[-1]  # Get the last date in the DataFrame
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=15)

# Forecast the next 15 days using the LSTM model
forecasted_prices = forecast_next_days(model, np.array(df['Close']), scaler, days=15)

# Create a DataFrame with the future dates and forecasted prices
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Price': forecasted_prices
})
forecast_df.set_index('Date', inplace=True)

# Display the forecast DataFrame
st.subheader('Forecasted Closing Prices for the Next 15 Days')
st.write(forecast_df)

# Plot the forecasted prices with the actual dates
st.subheader('Forecasted Closing Price Trend')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(forecast_df.index, forecast_df['Forecasted Price'], 'r', marker='o', label='Forecasted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('15-Day Forecasted Closing Price')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig3)


import base64# Add some vertical space before the logo
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

# Load the image and convert it to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Get the base64 string of the logo image
logo_base64 = get_base64_of_bin_file('logo.png')

# Display the logo in the bottom right corner using custom HTML and CSS
st.markdown(
    f"""
    <style>
    .logo-container {{
        position: fixed;
        bottom: 10px;
        right: 180px;
        width: 150px; /* Adjust the size as needed */
        z-index: 100;
    }}
    </style>
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)