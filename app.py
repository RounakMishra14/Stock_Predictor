import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

start='2010-01-01'
end='2024-10-04'

st.title('Stock Prediction')

user_input= st.text_input('Enter Stock Ticker', 'TSLA')
df=yf.download(user_input, start=start, end=end)

st.subheader('Summary Stats of the Stock:')
st.write(df.describe())

st.subheader('Closing Price Vs Time')
fig= plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('MA100 with MA200')
fig= plt.figure(figsize=(12, 6))
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
plt.plot(df.Close)
plt.plot(ma100, color='r')
plt.plot(ma200, color='g')
st.pyplot(fig)

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
        right: 100px;
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