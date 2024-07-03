import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima(data) :

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    
    train_size = int(len(data) * 0.8)
    train = data[:train_size]
    test = data[train_size:]

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=len(test))

    # st.write(forecast)
    return forecast