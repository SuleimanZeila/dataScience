import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2021-12-31'

st.set_page_config(
   page_title="Networkia Stock Prediction APP",
   page_icon="🧊"
)

st.title("Networkia Stock Trend Prediction")
user_input = st.text_input("Enter Stock Ticker",'AAPL')

df = data.DataReader(user_input,'yahoo', start, end)

#decribing Data

st.subheader("Data From 2010 - 2021")
st.write(df.describe())


#VIRTULIZATIONS
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100MA')
m100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(m100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100MA & 200MA')
m100 = df.Close.rolling(100).mean()
m200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(m100)
plt.plot(m200)
plt.plot(df.Close)
st.pyplot(fig)


####splitting  Data into Training and Testing 

data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])

data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#load my model
model = load_model("keras_model.h5")

pass_100_days = data_training.tail(100)
final_df = pass_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test =[]
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test,y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)



scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#final gragh
st.subheader('Predictions Vs Original')
fig2=plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)