%%writefile app.py
import streamlit as st
import tempfile
import cv2   
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning) 
import math  
import pandas as pd
import yfinance as yf


import matplotlib.pyplot as plt
import seaborn as sns
import urllib
#import requests
#from datetime import datetime
from keras.models import load_model
from keras.models import Sequential
# import Image from pillow to open images
from PIL import Image
import pickle

import plotly.express as px
import datetime as dt
from streamlit import caching
import plotly.graph_objects as go
from plotly.subplots import make_subplots
st.title("DEPLOYMENT")


st.title("DEPLOYMENT")
st.markdown("<h1 style='text-align: left; color: blue;'>Trading\n</h1>"

            "<h1 style='text-align: centre; color: red;'>Predicting Stock Prices</h1>",
            unsafe_allow_html=True)
st.write("STOCK MARKET PREDICTION AND ANALYSIS--")


def load_data():
    data = load_model('/content/drive/My Drive/Colab Notebooks/kbs/my_model.h5')
    #data.reset_index(inplace=True)
    df = pd.DataFrame(list(data.values()))
    
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=data.Date, y=data['Open'], name="stock_open",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(
        x=df.Date, y=df['Close'], name="stock_close", line_color='deepskyblue'))
    fig.layout.update(
        title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)




# Model

# imports

    data_load_state = st.text('Loading Model...')

    data = df.filter(['Close'])
    current_data = np.array(data).reshape(-1, 1).tolist()


    df = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_df = scaler.fit_transform(np.array(df).reshape(-1, 1))
    train_data = scaled_df[0:, :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_df[-60:, :].tolist()
    x_test = []
    y_test = []
    for i in range(60, 70):
        x_test = (test_data[i-60:i])
        x_test = np.asarray(x_test)
        pred_data = model.predict(x_test.reshape(1, x_test.shape[0], 1).tolist())

        y_test.append(pred_data[0][0])
        test_data.append(pred_data)


    pred_next_10 = scaler.inverse_transform(np.asarray(y_test).reshape(-1, 1))

    data_load_state.text('Loading Model... done!')


    st.subheader("Next 10 Days")
    st.write(pred_next_10)


    # pred = current_data.extend(pred_next_10.tolist())


    # plt.figure(figsize=(16, 8))
    # plt.title('model')
    # plt.xlabel('Date', fontsize=18)
    # plt.ylabel('Close_Price', fontsize=18)
    # plt.plot(pred)
    # plt.legend(['train'], loc='lower right')
    # plt.show()
load_data()
