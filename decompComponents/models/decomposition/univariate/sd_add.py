from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import streamlit as st

def seasonal_decompose_additive(selected_node_feature_data) :
    
    res = seasonal_decompose(selected_node_feature_data, model='additive')
    res.plot()
    plt.show()
    st.pyplot(plt)

    seasonal , trend , residue = res.seasonal, res.trend, res.resid

    # remove first 3 and last 3 values as they are NAN
    seasonal = seasonal[3:-3]
    trend = trend[3:-3]
    residue = residue[3:-3]

    return seasonal, trend, residue