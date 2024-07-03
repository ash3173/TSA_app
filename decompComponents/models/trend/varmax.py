from statsmodels.tsa.statespace.varmax import VARMAX
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import streamlit as st

from tradcomponents.functions import rmse

def varmax(df):
    # Split the dataframe into train and test sets
    train_size = int(len(df) * 0.8)
    train, test = df[0:train_size], df[train_size:len(df)]
    
    # Fit the VARMAX model
    model = VARMAX(train, order=(1, 1))
    fitted_model = model.fit(disp=False)

    # Generate a forecast
    forecast = fitted_model.forecast(steps=len(test))
    # forecast = pd.DataFrame(forecast, index=test.index, columns=test.columns)

    st.write("Forecasted values")   
    st.write(forecast)
    return forecast