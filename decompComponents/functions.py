from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from statsmodels.tsa.seasonal import STL

def get_node_data_from_merged(merged_data, node_index):
    filtered = merged_data[merged_data["node"] == node_index]
    filtered.drop(["node"], axis=1, inplace=True)
    unflattened = unflatten_dataframe(filtered)
    return unflattened

def unflatten_dataframe(df_flat):
    df = df_flat.pivot(index='timestamp', columns='feature', values='value')
    start_date = datetime.today().date() - timedelta(days=len(df))
    df["timestamp"] = pd.date_range(start=start_date, periods=len(df), freq='D')
    df.set_index('timestamp', inplace=True)
    return df

def get_feature_with_dates(df,feature_index) :    
    # filtered = df.iloc[:,feature_index:feature_index+1]
    filtered = df.iloc[:,feature_index]
    return filtered

def decompose_multivariate(selected_node_data) :

    trend_data = selected_node_data.copy()
    seasonal_data = selected_node_data.copy()
    residue_data = selected_node_data.copy()

    # Preprocessing the data
    for i in range(len(selected_node_data.columns)) :
        selected_node_feature_data = get_feature_with_dates(df=selected_node_data, feature_index=i)
        
        stl = STL(selected_node_feature_data,robust=True,)
        res = stl.fit()

        seasonal,trend,residue = res.seasonal, res.trend, res.resid
        trend_data.iloc[:,i] = trend
        seasonal_data.iloc[:,i] = seasonal
        residue_data.iloc[:,i] = residue

    return trend_data, seasonal_data, residue_data