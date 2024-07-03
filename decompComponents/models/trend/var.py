from statsmodels.tsa.vector_ar.var_model import VAR
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import streamlit as st


def var(df):

    # Split the dataframe into train and test sets
    train_size = int(len(df) * 0.8)
    train, test = df[0:train_size], df[train_size:len(df)]
    
    # Fit the VAR model
    model = VAR(train)
    fitted_model = model.fit()

    # Generate a forecast
    forecast = fitted_model.forecast(train.values, steps=len(test))

    # convert forecast to series dataframe
    forecast = pd.DataFrame(forecast, index=test.index, columns=test.columns)

    return forecast
