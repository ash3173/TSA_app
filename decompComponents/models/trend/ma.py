import streamlit as st
import statsmodels.api as sm

def ma(data) :
    train_size = int(len(data) * 0.8)
    train = data[:train_size]
    test = len(data) - train_size

    model = sm.tsa.arima.ARIMA(train, order=(0, 0, 3))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=test)

    # st.write(forecast)
    return forecast