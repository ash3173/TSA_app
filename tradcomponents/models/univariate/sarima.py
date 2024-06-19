# sarima_model_file.py
import streamlit as st
# sarima_model_file.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

def get_node_data_from_merged(merged_data, node_index):
    filtered = merged_data[merged_data["node"] == node_index].copy()  # Ensure to work on a copy
    filtered.drop(["node"], axis=1, inplace=True)
    unflattened = unflatten_dataframe(filtered)
    return unflattened

def unflatten_dataframe(df_flat):
    df = df_flat.pivot(index='timestamp', columns='feature', values='value')
    df.reset_index(drop=True, inplace=True)
    df.columns.name = None
    return df

def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna(), autolag='AIC')
    p_value = result[1]
    return p_value < 0.05

def fit_sarima_forecast(timeseries, order, seasonal_order, forecast_periods):
    train = timeseries[:-forecast_periods]
    test = timeseries[-forecast_periods:]

    # Fit SARIMA model
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Forecast
    forecast = model_fit.get_forecast(steps=forecast_periods)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(timeseries.index[-100:], timeseries[-100:], label='Actual', linewidth=3, c='blue')
    plt.plot(forecast_mean.index, forecast_mean, label='Forecast', linestyle='--', c='red')
    plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.2)
    plt.title('Actual vs Forecast (SARIMA)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    return plt.gcf()  # Return the current figure object for Streamlit

def run_sarima_model(node_data, edge_data):
    node_index = 0  # Replace with appropriate node index selection logic
    timeseries = get_node_data_from_merged(node_data, node_index)
    
    for feature in timeseries.columns:
        print(f'\nFeature: {feature}')
        series = timeseries[feature]
    
        if not check_stationarity(series):
            print('Applying differencing...')
            series_diff = series.diff().dropna()
            is_stationary = check_stationarity(series_diff)
        else:
            print('Series is stationary.')
            is_stationary = True
    
        if is_stationary:
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 12)
            forecast_periods = 12
    
            fig = fit_sarima_forecast(series, order, seasonal_order, forecast_periods)
            st.pyplot(fig)  # Display plot using Streamlit
