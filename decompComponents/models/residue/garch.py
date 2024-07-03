from arch import arch_model
import pandas as pd
import streamlit as st

def garch(data, p=1, q=1):
    test_size = int(len(data) * 0.2)  # Using the last 20% for rolling forecast
    train_size = len(data) - test_size
    forecast_series = pd.Series(index=data[-test_size:].index)

    for i in range(train_size, len(data)):
        train = data[:i]
        
        model = arch_model(train, vol='Garch', p=p, q=q)
        model_fit = model.fit(disp='off')
        
        forecast = model_fit.forecast(horizon=1)
        forecast_mean = forecast.mean.iloc[-1]['h.1']
        
        forecast_series.loc[data.index[i]] = forecast_mean

    forecast_series = forecast_series.rename("predicted_mean")
        
    return forecast_series