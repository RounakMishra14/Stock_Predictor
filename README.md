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
```
scaler = MinMaxScaler(feature_range=(0, 1))
data_trainning_array = scaler.fit_transform(data_trainning)
data_trainning_array
```
The training data is scaled to a range between 0 and 1 using MinMaxScaler. Normalizing the data helps the LSTM model converge faster and perform better since it reduces the variance and scales all values uniformly.
```
x_train = []
y_train = []
for i in range(100, data_trainning_array.shape[0]):
    x_train.append(data_trainning_array[i-100:i])
    y_train.append(data_trainning_array[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
```
The training dataset is structured by creating sequences of 100 days (x_train) to predict the next day’s closing price (y_train).
* Sliding Window: A window of 100 consecutive closing prices is used to predict the price at the next time step.
* Feature and Target Arrays: The feature array (x_train) holds sequences of the past 100 days, while the target array (y_train) contains the next day's price for each sequence.
  
The x_train and y_train arrays are then converted into NumPy arrays, making them compatible with the LSTM model’s input requirements.

## 🧠 LSTM Model Building
The LSTM model is designed to predict stock prices based on historical closing prices.

Model Architecture:

The LSTM model comprises multiple layers:
* Four LSTM layers with dropout layers in between to prevent overfitting.
* A dense layer at the end to output a single prediction value.

```
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
```

Training:
* The model is compiled using the adam optimizer and mean_squared_error as the loss function.
It is trained for 50 epochs, with the performance evaluated using the testing set.

Forecasting:
* The model forecasts the next 15 days of closing prices based on user-selected stock data.

## 🖥️ App Features

Chart Visualization:

![An interactive chart displays stock price movements.
Users can zoom in/out using a date range slider for detailed exploration.](1.png)

