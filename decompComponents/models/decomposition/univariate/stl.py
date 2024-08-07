import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import plotly_express as px
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.statespace.sarimax import SARIMAX


def stl_model(f1) :

    stl = STL(f1,robust=True,)
    res = stl.fit()
    fig = res.plot()
    plt.show()
    st.pyplot(plt)

    return res.seasonal, res.trend, res.resid


    # train_size = int(len(f1) * 0.8)
    # test_size = len(f1) - train_size

    # stlf = STLForecast(f1[:train_size], SARIMAX, model_kwargs=dict(order=(3, 1, 3), trend="t"),period=14)
    # stlf_res = stlf.fit()
    # forecast = stlf_res.forecast(test_size)

    # forecast = forecast.to_frame(name='predicted_value')

    # final = pd.concat([f1, forecast], axis=1)
    # plot = px.line(final, x=final.index, y=final.columns, title="Forecast using STL decomposition model using SARIMA")
    # st.plotly_chart(plot)
    
    
    # train_size = int(len(f1) * 0.8)
    # test_size = len(f1) - train_size

    # stlf = STLForecast(f1[:train_size], ARIMA, model_kwargs=dict(order=(3, 1, 3), trend="t"),period=14)
    # stlf_res = stlf.fit()
    # forecast = stlf_res.forecast(test_size)

    # forecast = forecast.to_frame(name='predicted_value')

    # final = pd.concat([f1, forecast], axis=1)
    # plot = px.line(final, x=final.index, y=final.columns, title="Forecast using STL decomposition model using ARIMA")
    # st.plotly_chart(plot)

    