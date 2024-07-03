import streamlit as st
import statsmodels.api as sm

def arima(data) :
    train_size = int(len(data) * 0.8)
    train = data[:train_size]
    test = len(data) - train_size

    model = sm.tsa.arima.ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=test)

    return forecast