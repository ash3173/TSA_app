import pandas as pd
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import plotly_express as px

from statsmodels.tsa.stattools import acf, pacf  # For ACF and PACF calculations
import numpy as np  # For array operations


def arima_model(data) :

    result = adfuller(data)
    st.write("ADF Statistic: ", result[0])
    st.write("p-value: %f", result[1])

    acf_plot = plot_acf(data)
    st.pyplot(acf_plot)

    pacf_plot = plot_pacf(data)
    st.pyplot(pacf_plot)

    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size

    model = sm.tsa.arima.ARIMA(data[:train_size], order=(8,1,5))
    model_fit = model.fit()
    forecast = model_fit.forecast(test_size)

    final = pd.concat([data, forecast], axis=1)

    plot = px.line(final, x=final.index, y=final.columns, title="Forecast using ARIMA model")
    st.plotly_chart(plot)