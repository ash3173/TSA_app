import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import matplotlib.pyplot as plt

from decompComponents.functions import get_node_data_from_merged , get_feature_with_dates

from decompComponents.models.trend.arima import arima
from decompComponents.models.trend.ma import ma
from decompComponents.models.trend.var import var
from decompComponents.models.trend.varmax import varmax

from decompComponents.models.seasonal.sarima import sarima

from decompComponents.models.residue.garch import garch
from decompComponents.models.residue.arch import arch

from decompComponents.models.decomposition.univariate.stl import stl_model
from decompComponents.models.decomposition.univariate.sd_add import seasonal_decompose_additive
from decompComponents.models.decomposition.univariate.sd_mul import seasonal_decompose_multiplicative

from decompComponents.models.decomposition.multivariate.stl_multivariate import decompose_multivariate
from decompComponents.models.decomposition.multivariate.seasonal_decompose_mult import decompose_multivariate_seasonal

from decompComponents.models.deepLearning.multivariate.new_models.multi_lstm import multi_LSTM
from decompComponents.models.deepLearning.multivariate.new_models.multi_gru import multi_GRU
from decompComponents.models.deepLearning.multivariate.new_models.multi_cnn import multi_CNN
from decompComponents.models.deepLearning.multivariate.new_models.multi_cnn_gru import multi_CNN_GRU
from decompComponents.models.deepLearning.multivariate.new_models.multi_cnn_lstm import multi_CNN_LSTM

from tradcomponents.functions import rmse
from tradcomponents.functions import differencing



st.title("Time Series Analysis using Decomposition.")

node = st.sidebar.file_uploader("Upload a Node data file", type=["csv"])
edge = st.sidebar.file_uploader("Upload a Edge data file", type=["csv"])

if node and edge :
    st.write("Files uploaded successfully")

    node_data = pd.read_csv(node)
    edge_data = pd.read_csv(edge)

    num_nodes = len(node_data["node"].unique())
    node_index = st.slider("Select the node index", 0, num_nodes - 1)

    temp = node_data
    st.write("merged data for node", node_index)
    selected_node_data = get_node_data_from_merged(merged_data=temp, node_index=node_index)
    st.write(selected_node_data)

    options = ["Traditional","Deep Learning"]
    option = st.selectbox("Choose a model",options)

    if option == "Traditional":
            
        options_varite = ["Select","UniVariate","MultiVariate"]
        option_varite = st.selectbox("Choose a model",options_varite)

        if option_varite == "UniVariate" :

            num_features = len(node_data["feature"].unique())
            feature_index = st.slider("Select the feature index", 0, num_features - 1)

            selected_node_feature_data = get_feature_with_dates(df=selected_node_data, feature_index=feature_index)
            
            decompose_options = ["Select", "STL","Seasonal Decompose "]
            model_option = st.selectbox("Choose a Decomposition method", decompose_options)

            if model_option == "STL":
                seasonal,trend,residue = stl_model(selected_node_feature_data) 
            elif model_option == "Seasonal Decompose " :
                seasonal,trend,residue = seasonal_decompose_additive(selected_node_feature_data)


            options = ["Select","Sarima","ARIMA","MA", "ARCH","Garch" ]
            # seasonal_options = ["Select","Sarima","ARIMA","MA"]
            # trend_options = ["Select","Sarima","ARIMA","MA"]
            # residue_options = ["Select","Sarima","ARIMA","MA"]

            seasonal_option = st.selectbox("Choose a seasonal model",options)
            trend_option = st.selectbox("Choose a trend model",options)
            residue_option = st.selectbox("Choose a residue model",options)

            if seasonal_option == "Sarima" :
                forecasted_seasonal = sarima(seasonal)
            elif seasonal_option == "ARIMA" :
                forecasted_seasonal = arima(seasonal)
            elif seasonal_option == "MA" :
                forecasted_seasonal = ma(seasonal)
            elif seasonal_option == "Garch" :
                forecasted_seasonal = garch(seasonal)
            elif seasonal_option == "ARCH" :
                forecasted_seasonal = arch(seasonal)
            

            if trend_option == "Sarima" :
                forecasted_trend = sarima(trend)
            elif trend_option == "ARIMA" : #Arima is not able to capture trend
                forecasted_trend = arima(trend)
            elif trend_option == "MA" : #MA is not able to capture trend
                forecasted_trend = ma(trend)
            elif trend_option == "Garch" :
                forecasted_trend = garch(trend)
            elif trend_option == "ARCH" :
                forecasted_trend = arch(trend)
            
            
            if residue_option == "Sarima" :
                forecasted_residue = sarima(residue)
            elif residue_option == "ARIMA" :
                forecasted_residue = arima(residue)
            elif residue_option == "MA" :
                forecasted_residue = ma(residue)
            elif residue_option == "Garch" :
                forecasted_residue = garch(residue)
            elif residue_option == "ARCH" :
                forecasted_residue = arch(residue)


            if model_option != "Select" and residue_option != "Select" and seasonal_option != "Select" and trend_option != "Select":
                forecast = forecasted_seasonal + forecasted_trend + forecasted_residue
                # forecast = forecasted_seasonal * forecasted_trend * forecasted_residue
                final = pd.concat([selected_node_feature_data,forecast],axis=1)

                fig = px.line(final, x=final.index, y=final.columns , title="Forecasting using both seasonality and trend")
                st.plotly_chart(fig)

            test_size = int(len(selected_node_feature_data) * 0.2)
            actual_y = selected_node_feature_data.iloc[-test_size:]

            button = st.button("Choose the best model")
            if button:
                best = []
                diff_best = []
                error = float("inf")
                diff_error = float("inf")

                for i in options :
                    for j in options :
                        for k in options :
                            
                            if i != "Select" and j != "Select" and k != "Select" :

                                if i == "Sarima" :
                                    forecasted_seasonal = sarima(seasonal)
                                elif i == "ARIMA" :
                                    forecasted_seasonal = arima(seasonal)
                                elif i == "MA" :
                                    forecasted_seasonal = ma(seasonal)
                                elif i == "ARCH" :
                                    forecasted_seasonal = arch(seasonal)
                                elif i == "Garch" :
                                    forecasted_seasonal = garch(seasonal)

                                if j == "Sarima" :
                                    forecasted_trend = sarima(trend)
                                elif j == "ARIMA" :
                                    forecasted_trend = arima(trend)
                                elif j == "MA" :
                                    forecasted_trend = ma(trend)
                                elif j == "ARCH" :
                                    forecasted_trend = arch(trend)
                                elif j == "Garch" :
                                    forecasted_trend = garch(trend)

                                if k == "Sarima" :
                                    forecasted_residue = sarima(residue)
                                elif k == "ARIMA" :
                                    forecasted_residue = arima(residue)
                                elif k == "MA" :
                                    forecasted_residue = ma(residue)
                                elif k == "ARCH" :
                                    forecasted_residue = arch(residue)
                                elif k == "Garch" : 
                                    forecasted_residue = garch(residue)

                                forecast = forecasted_seasonal + forecasted_trend + forecasted_residue
                                # forecast = forecasted_seasonal * forecasted_trend * forecasted_residue

                                error_value = rmse(actual_y,forecast)
                                diff_error_value = differencing(actual_y,forecast)
                                st.write("For",i,j,k,"Error value is ",diff_error_value)
                                
                                if error_value < error :
                                    best = [i,j,k,error_value]
                                    error = error_value

                                if diff_error_value < diff_error :
                                    diff_best = [i,j,k,diff_error_value]
                                    diff_error = diff_error_value

                st.write("Best model is ",best)
                st.write("Best model based on differencing actual y vs predicted y is ",diff_best)
            
        elif option_varite == "MultiVariate" :
                
            decompose_options = ["Select", "STL","Seasonal Decompose "]
            model_option = st.selectbox("Choose a Decomposition method", decompose_options)

            if model_option == "STL":
                trend_data , seasonal_data , residue_data = decompose_multivariate(selected_node_data=selected_node_data)
            elif model_option == "Seasonal Decompose" :
                trend_data , seasonal_data , residue_data = decompose_multivariate_seasonal(selected_node_data,model="additive")

            options = ["Select","VAR","Varmax"]
            seasonal_option = st.selectbox("Choose a seasonal model",options)
            trend_option = st.selectbox("Choose a trend model",options)
            residue_option = st.selectbox("Choose a residue model",options)

            # forecasting based on the selected models
            
            if seasonal_option == "VAR" :
                forecasted_seasonal = var(seasonal_data)
            elif seasonal_option == "Varmax" :
                forecasted_seasonal = varmax(seasonal_data)


            if trend_option == "VAR" :
                forecasted_trend = var(trend_data)
            elif trend_option == "Varmax" :
                forecasted_trend = varmax(trend_data)


            if residue_option == "VAR" :
                forecasted_residue = var(residue_data)
            elif residue_option == "Varmax" :
                forecasted_residue = varmax(residue_data)

            # plotting the model
            if model_option != "Select" and residue_option != "Select" and seasonal_option != "Select" and trend_option != "Select":

                test_size = int(len(selected_node_data) * 0.2)
                forecast = selected_node_data.copy()[-test_size:]

                for i in range(len(selected_node_data.columns)) :
                    forecast.iloc[:,i] = forecasted_seasonal.iloc[:,i] + forecasted_trend.iloc[:,i] + forecasted_residue.iloc[:,i]
                    forecast.rename(columns={forecast.columns[i]:f'forecasted_{i}'},inplace=True)

                final = pd.concat([selected_node_data,forecast],axis=1)
                fig = px.line(final, x=final.index, y=final.columns , title="Forecasting using both seasonality and trend")
                st.plotly_chart(fig)


            # Choosing the best model
            button = st.button("Choose the best model")
            if button:

                best = []
                error = float("inf")

                for i in options :
                    for j in options :
                        for k in options :
                            
                            if i != "Select" and j != "Select" and k != "Select" :

                                if i == "VAR" :
                                    forecasted_seasonal = var(seasonal_data)
                                elif i == "Varmax" :
                                    forecasted_seasonal = varmax(seasonal_data)

                                if j == "VAR" :
                                    forecasted_trend = var(trend_data)
                                elif j == "Varmax" :
                                    forecasted_trend = varmax(trend_data)

                                if k == "VAR" :
                                    forecasted_residue = var(residue_data)
                                elif k == "Varmax" :
                                    forecasted_residue = varmax(residue_data)

                                forecast = selected_node_data.copy()[-test_size:]

                                for l in range(len(selected_node_data.columns)) :
                                    forecast.iloc[:,l] = forecasted_seasonal.iloc[:,l] + forecasted_trend.iloc[:,l] + forecasted_residue.iloc[:,l]

                                error_value = rmse(selected_node_data.copy()[-test_size:],forecast)
                                st.write("For",i,j,k,"Error value is ",error_value)
                                
                                if error_value < error :
                                    best = [i,j,k,error_value]
                                    error = error_value

                st.write("Best model is ",best)


    elif option == "Deep Learning":

        feature_headers = selected_node_data.columns.tolist()
        selected_feature_headers = st.multiselect("Select feature headers for multivariate analysis", feature_headers, default=[feature_headers[0]])

        selected_node_features_data = selected_node_data[selected_feature_headers]
        st.write(selected_node_features_data)

        forecast_headers = st.multiselect("Select feature headers for forecasting", selected_feature_headers, default=selected_feature_headers[:2])
        
        if not forecast_headers:
            st.warning("Please select at least one feature header for forecasting.")
        else:
            selected_forecast_features_data = selected_node_features_data[forecast_headers] #this is the actual data that needed to be ploted
            st.write("Selected features for forecasting:")
            st.write(selected_forecast_features_data)
        
            decompose_options = ["Select", "STL","Seasonal Decompose"]
            model_option = st.selectbox("Choose a Decomposition method", decompose_options)

            if model_option == "STL":
                trend_features_data , seasonal_features_data , residue_features_data = decompose_multivariate(selected_node_data=selected_node_features_data)
            elif model_option == "Seasonal Decompose" :
                trend_features_data , seasonal_features_data , residue_features_data = decompose_multivariate_seasonal(selected_node_data=selected_node_features_data,model="additive")

            options = ["Select","LSTM","GRU","CNN","CNN-LSTM","CNN-GRU"]
            seasonal_option = st.selectbox("Choose a seasonal model",options)
            trend_option = st.selectbox("Choose a trend model",options)
            residue_option = st.selectbox("Choose a residue model",options)

            st.markdown("""
            <style>
            .big-font {
                font-size:25px !important;
            }
            </style>
            """, unsafe_allow_html=True)


            st.markdown('<p class="big-font">Forecasting for Seasonal Data </p>', unsafe_allow_html=True)

            if seasonal_option == "LSTM" :
                forecasted_seasonal = multi_LSTM(seasonal_features_data,forecast_headers,key=0)
            elif seasonal_option == "GRU" :
                forecasted_seasonal = multi_GRU(seasonal_features_data,forecast_headers,key=0)
            elif seasonal_option == "CNN" :
                forecasted_seasonal = multi_CNN(seasonal_features_data,forecast_headers,key=0)
            elif seasonal_option == "CNN-LSTM" :
                forecasted_seasonal = multi_CNN_LSTM(seasonal_features_data,forecast_headers,key=0)
            elif seasonal_option == "CNN-GRU" :
                forecasted_seasonal = multi_CNN_GRU(seasonal_features_data,forecast_headers,key=0)
            


            st.markdown('<p class="big-font">Forecasting for Trend Data </p>', unsafe_allow_html=True)

            if trend_option == "LSTM" :
                forecasted_trend = multi_LSTM(trend_features_data,forecast_headers,key=1)
            elif trend_option == "GRU" :
                forecasted_trend = multi_GRU(trend_features_data,forecast_headers,key=1)
            elif trend_option == "CNN" :
                forecasted_trend = multi_CNN(trend_features_data,forecast_headers,key=1)
            elif trend_option == "CNN-LSTM" :
                forecasted_trend = multi_CNN_LSTM(trend_features_data,forecast_headers,key=1)
            elif trend_option == "CNN-GRU" :
                forecasted_trend = multi_CNN_GRU(trend_features_data,forecast_headers,key=1)



            st.markdown('<p class="big-font">Forecasting for Residue Data </p>', unsafe_allow_html=True)

            if residue_option == "LSTM" :
                forecasted_residue = multi_LSTM(residue_features_data,forecast_headers,key=2)
            elif residue_option == "GRU" :
                forecasted_residue = multi_GRU(residue_features_data,forecast_headers,key=2)
            elif residue_option == "CNN" :
                forecasted_residue = multi_CNN(residue_features_data,forecast_headers,key=2)
            elif residue_option == "CNN-LSTM" :
                forecasted_residue = multi_CNN_LSTM(residue_features_data,forecast_headers,key=2)
            elif residue_option == "CNN-GRU" :
                forecasted_residue = multi_CNN_GRU(residue_features_data,forecast_headers,key=2)


            
            # plotting the model
            if model_option != "Select" and residue_option != "Select" and seasonal_option != "Select" and trend_option != "Select":
                
                test_size = int(len(selected_forecast_features_data) * 0.1)
                train_size = len(selected_forecast_features_data) - test_size

                # plt.plot(selec)

                test_data_with_headers = selected_forecast_features_data[forecast_headers]
                st.write("Actual Data",test_data_with_headers)

                forecast = test_data_with_headers.copy()[-test_size:]
                forecast_without_residue = test_data_with_headers.copy()[-test_size:]
                
                for i in range(len(test_data_with_headers.columns)) :
                    forecast.iloc[:,i] = forecasted_seasonal.iloc[:,i] + forecasted_trend.iloc[:,i] + forecasted_residue.iloc[:,i]
                    forecast_without_residue.iloc[:,i] = forecasted_seasonal.iloc[:,i] + forecasted_trend.iloc[:,i] 
                    forecast.rename(columns={forecast.columns[i]:f'forecasted_{i}'},inplace=True)
                    forecast_without_residue.rename(columns={forecast_without_residue.columns[i]:f'forecasted_{i}'},inplace=True)
                
                final = pd.concat([test_data_with_headers,forecast],axis=1)
                fig = px.line(final, x=final.index, y=final.columns , title="Combined Forecast")
                st.plotly_chart(fig)
                
                deResidued_Data = test_data_with_headers - residue_features_data
                final_deResidue = pd.concat([deResidued_Data,forecast_without_residue],axis=1)
                # fig = px.line(final_deResidue, x=final_deResidue.index, y=final_deResidue.columns , title="Forecast without Residue")
                # st.plotly_chart(fig)