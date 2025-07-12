import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st

import os  # Import the 'os' module for path manipulation

model_path = 'C:\\Users\\presh\\OneDrive\\Desktop\\BNB\\Stock Prediction Model.keras'
encoded_path = os.path.abspath(model_path.encode('utf-8')).decode('utf-8')
model = load_model(encoded_path)




from PIL import Image


# Load the logo image
logo_image = open(r'C:\Users\presh\Downloads\logo.png', 'rb').read()

# Display the logo image in the sidebar
st.sidebar.image(logo_image, caption='BULLS AND BEARS', use_column_width=True)


# Using HTML/CSS within a markdown string to change the font
st.markdown('<h1 style="font-family: Times New Roman; font-size: 40px; font-weight: bold;">Stock Market Predictor</h1>', unsafe_allow_html=True)


st.markdown("""
    <style>
        /* CSS to change the font size of the label */
        .stTextInput label {
            font-size: 35px; /* Adjust the font size as needed */
        }

        /* CSS to change the font size of the text input */
        .stTextInput>div>div>input {
            font-size: 15px; /* Adjust the font size as needed */
        }
    </style>
""", unsafe_allow_html=True)
stock=st.text_input('Enter Stock Market Symbol','GOOG')
start=st.text_input('Enter Start Date(YYYY-MM-DD)','2012-01-01')
end=st.text_input('Enter End Date(YYYY-MM-DD)','2024-01-01')

data = yf.download(stock, start ,end)

if data.empty:
    st.error("No data found! Please check the stock symbol or date range.")
    st.stop()

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test= pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)


ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,8))
plt.plot(ma_50_days, 'r',label='Moving Average 50')
plt.plot(data.Close, 'g',label='Closing Point')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Price', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()



ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(10,8))
plt.plot(ma_50_days, 'r',label='Moving Average 50')
plt.plot(ma_100_days, 'b',label='Moving Average 100')
plt.plot(data.Close, 'g',label='Closing Point')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Price', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()



ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(10,8))
plt.plot(ma_100_days, 'r',label='Moving Average 100')
plt.plot(ma_200_days, 'b',label='Moving Average 200')
plt.plot(data.Close, 'g')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Price', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()



x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

import os  # Import the 'os' module for path manipulation

predict=model.predict(x)
scale=1/scaler.scale_
predict= predict*scale
y=y*scale


fig4 = plt.figure(figsize=(10,8))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'b', label='Predicted Price')
plt.xlabel('Time', fontsize=18, fontname='Times New Roman')
plt.ylabel('Price', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Dropdown for selecting the plot type
plot_type = st.selectbox("Select Plot", ['Price vs Moving Average 50',
                                         'Price vs Moving Average 50 vs Moving Average 100',
                                         'Price vs Moving Average 100 vs Moving Average 200',
                                         'Original Price vs Predicted Price'])

# Generate the selected plot
if plot_type == 'Price vs Moving Average 50':
    st.subheader('Price vs Moving Average 50')
    ma_50_days = data.Close.rolling(50).mean()
    fig = plt.figure(figsize=(10, 8))
    plt.plot(ma_50_days, 'r', label='Moving Average 50')
    plt.plot(data.Close, 'g', label='Closing Point')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Price', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    st.pyplot(fig)
elif plot_type == 'Price vs Moving Average 50 vs Moving Average 100':
    st.subheader('Price vs Moving Average 50 vs Moving Average 100')
    ma_100_days = data.Close.rolling(100).mean()
    fig = plt.figure(figsize=(10, 8))
    plt.plot(ma_50_days, 'r', label='Moving Average 50')
    plt.plot(ma_100_days, 'b', label='Moving Average 100')
    plt.plot(data.Close, 'g', label='Closing Point')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Price', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    st.pyplot(fig)
elif plot_type == 'Price vs Moving Average 100 vs Moving Average 200':
    st.subheader('Price vs Moving Average 100 vs Moving Average 200')
    ma_200_days = data.Close.rolling(200).mean()
    fig = plt.figure(figsize=(10, 8))
    plt.plot(ma_100_days, 'r', label='Moving Average 100')
    plt.plot(ma_200_days, 'b', label='Moving Average 200')
    plt.plot(data.Close, 'g', label='Closing Point')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Price', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    st.pyplot(fig)
elif plot_type == 'Original Price vs Predicted Price':
    st.subheader('Original Price vs Predicted Price')
    x = []
    y = []
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i - 100:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x), np.array(y)

    predict = model.predict(x)
    scale = 1 / scaler.scale_
    predict = predict * scale
    y = y * scale

    fig = plt.figure(figsize=(10, 8))
    plt.plot(predict, 'r', label='Original Price')
    plt.plot(y, 'b', label='Predicted Price')
    plt.xlabel('Time', fontsize=18, fontname='Times New Roman')
    plt.ylabel('Price', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    st.pyplot(fig)










