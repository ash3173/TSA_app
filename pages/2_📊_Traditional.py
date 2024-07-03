from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly_express as px
from tradcomponents.functions import get_node_data_from_merged, get_feature_with_dates
from tradcomponents.models.univariate.arima import arima_model
from tradcomponents.models.univariate.stl import stl_model
import numpy as np

from tradcomponents.models.univariate.prophet import prophet_forecast
from tradcomponents.models.univariate.expon_smoth import ets_forecast
from tradcomponents.models.univariate.arch import run_arch_model
from tradcomponents.models.univariate.garch import run_garch_model
from tradcomponents.models.univariate.sarima import run_sarima_model
from tradcomponents.models.univariate.sarimax import run_sarimax_model
from tradcomponents.models.univariate.neural_prophet import run_neural_prophet_model


from tradcomponents.models.multivariate.mul_prophet import multivariate_prophet_forecast
from tradcomponents.models.multivariate.mul_var import multivariate_var_forecast
from tradcomponents.models.multivariate.mult_varmax import multivariate_varmax_forecast


import streamlit as st
import warnings
warnings.filterwarnings("ignore")

st.title("Time Series Analysis using Traditional Methods.")

node, edge = None, None
node = st.sidebar.file_uploader("Upload a Node data file", type=["csv"])
edge = st.sidebar.file_uploader("Upload a Edge data file", type=["csv"])

forecast_type_options = ["Univariate",
                         "Multivariate - Node Level", 
                         #"Multivariate K-hop"
                         ]

selected_forecast_option = st.sidebar.selectbox(
    "Select the model", forecast_type_options)


if node and edge:
    node_data = pd.read_csv(node)
    edge_data = pd.read_csv(edge)

    st.write("Files uploaded successfully")

    if selected_forecast_option == "Univariate":

        options = ["Select","Exponential Smoothing", "ARIMA","SARIMA",
        #"SARIMAX", 
        "STL", 
        #"ARCH", "GARCH", 
        "Prophet","NeuralProphet"]
        selected_option = st.selectbox("Select the model", options)

        num_nodes = len(node_data["node"].unique())

        if selected_option == "ARCH":
            run_arch_model(node_data, edge_data)
        if selected_option == "GARCH":
            run_garch_model(node_data, edge_data)
        if selected_option == "SARIMA":
            run_sarima_model(node_data, edge_data)
        if selected_option == "SARIMAX":
            run_sarimax_model(node_data, edge_data)
        
        if selected_option == "ARIMA":
            num_features = len(node_data["feature"].unique())
            node_index = st.slider("Select the node index", 0, num_nodes - 1)

            temp = node_data
            st.write("merged data for node", node_index)
            selected_node_data = get_node_data_from_merged(merged_data=temp, node_index=node_index)
            st.write(selected_node_data)

            feature_index = st.slider("Select the feature index", 0, num_features - 1)
            selected_node_feature_data = get_feature_with_dates(df=selected_node_data, feature_index=feature_index)

            arima_model(data=selected_node_feature_data)
            

        if selected_option == "STL":
            num_nodes = len(node_data["node"].unique())
            num_features = len(node_data["feature"].unique())
            node_index = st.slider("Select the node index", 0, num_nodes - 1)

            temp = node_data
            st.write("merged data for node", node_index)
            selected_node_data = get_node_data_from_merged(merged_data=temp, node_index=node_index)
            st.write(selected_node_data)

            feature_index = st.slider("Select the feature index", 0, num_features - 1)
            selected_node_feature_data = get_feature_with_dates(df=selected_node_data, feature_index=feature_index)

            stl_model(selected_node_feature_data)

        if selected_option == "Prophet":
            results = {}
            rmse_values = []
            merged_data = node_data

            for node_index_forecast in range(num_nodes):
                node_data_forecast = get_node_data_from_merged(
                    merged_data=merged_data, node_index=node_index_forecast)
                rmse, train, test, forecasts = prophet_forecast(
                    df=node_data_forecast)
                results[node_index_forecast] = {
                    'rmse': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
                rmse_values.append(rmse)

            avg_rmse = np.mean(rmse_values)
            st.metric(f'Average RMSE', avg_rmse)

            node_index = st.slider(
                label="Node ID", min_value=0, max_value=max(num_nodes-1, 1))
            node_data = get_node_data_from_merged(
                merged_data=merged_data, node_index=node_index)

            st.subheader("Node Data")
            st.dataframe(node_data)

            result = results[node_index]
            train = result['train']
            test = result['test']
            forecasts = result['forecasts']

            # Define a list of colors for the plots
            # Add more colors if needed
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train)), train[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts[column]['yhat'][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts[column]['yhat'][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

        if selected_option == "NeuralProphet":
            st.sidebar.write("Select a feature index to forecast")
            feature_index = st.number_input("Feature Index", min_value=0, max_value=len(node_data["feature"].unique())-1, step=1, value=1)
            run_neural_prophet_model(node_data, edge_data, feature_index)
        
        if selected_option == "Exponential Smoothing":

            results = {}
            rmse_values = []
            merged_data = node_data

            for node_index_forecast in range(num_nodes):
                node_data_forecast = get_node_data_from_merged(
                    merged_data=merged_data, node_index=node_index_forecast)
                rmse, train, test, forecasts = ets_forecast(
                    df=node_data_forecast)
                results[node_index_forecast] = {
                    'rmse ': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
                rmse_values.append(rmse)

            rmse_mape = np.mean(rmse_values)
            st.metric(f'Average RMSE', rmse_mape)

            node_index = st.slider(
                label="Node ID", min_value=0, max_value=max(num_nodes-1, 1), key="ets")
            node_data = get_node_data_from_merged(
                merged_data=merged_data, node_index=node_index)

            st.subheader("Node Data")
            st.dataframe(node_data)

            result = results[node_index]
            train = result['train']
            test = result['test']
            forecasts = result['forecasts']

            # Define a list of colors for the plots
            # Add more colors if needed
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train)), train[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts[column][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Training, Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts[column][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

        # if selected_option == "ARCH":
        #     num_features = len(node_data["feature"].unique())
        #     node_index = st.slider("Select the node index", 0, num_nodes - 1)

        #     temp = node_data
        #     st.write("merged data for node", node_index)
        #     selected_node_data = get_node_data_from_merged(
        #         merged_data=temp, node_index=node_index)
        #     st.write(selected_node_data)

        #     plot = arch(selected_node_data)

        # if selected_option == "GARCH":
        #     num_features = len(node_data["feature"].unique())
        #     node_index = st.slider("Select the node index", 0, num_nodes - 1)

        #     temp = node_data
        #     st.write("merged data for node", node_index)
        #     selected_node_data = get_node_data_from_merged(
        #         merged_data=temp, node_index=node_index)
        #     st.write(selected_node_data)

        #     plot = Garch(selected_node_data)

    if selected_forecast_option == "Multivariate - Node Level":

        num_nodes = len(node_data["node"].unique())
        forecast_options = ["Select", "Multivariate Prophet",
                            "Multivariate VAR", "Multivariate VARMAX"]
        selected_option = st.selectbox("Select the model", forecast_options)
        merged_data = node_data

        if selected_option == "Multivariate Prophet":
            results = {}
            rmse_values = []

            for node_index_forecast in range(num_nodes):
                node_data_forecast = get_node_data_from_merged(
                    merged_data=merged_data, node_index=node_index_forecast)
                rmse, train, test, forecasts = multivariate_prophet_forecast(
                    df=node_data_forecast)
                results[node_index_forecast] = {
                    'rmse': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
                rmse_values.append(rmse)

            avg_rmse = np.mean(rmse_values)
            st.metric(f'Average RMSE', avg_rmse)

            node_index = st.slider(
                label="Node ID", min_value=0, max_value=max(num_nodes-1, 1))
            node_data = get_node_data_from_merged(
                merged_data=merged_data, node_index=node_index)

            st.subheader("Node Data")
            st.dataframe(node_data)

            result = results[node_index]
            train = result['train']
            test = result['test']
            forecasts = result['forecasts']

            # Define a list of colors for the plots
            # Add more colors if needed
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train)), train[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts[column]['yhat'][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Training, Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts[column]['yhat'][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

        if selected_option == "Multivariate VAR":
            results = {}
            rmse_values = []

            for node_index_forecast in range(num_nodes):
                node_data_forecast = get_node_data_from_merged(
                    merged_data=merged_data, node_index=node_index_forecast)
                rmse, train, test, forecasts = multivariate_var_forecast(
                    df=node_data_forecast)
                results[node_index_forecast] = {
                    'rmse': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
                rmse_values.append(rmse)

            avg_rmse = np.mean(rmse_values)
            st.metric(f'Average RMSE', avg_rmse)

            node_index = st.slider(label="Node ID", min_value=0, max_value=max(
                num_nodes-1, 1), key='mul_var')
            node_data = get_node_data_from_merged(
                merged_data=merged_data, node_index=node_index)

            st.subheader("Node Data")
            st.dataframe(node_data)

            result = results[node_index]
            train = result['train']
            test = result['test']
            forecasts = result['forecasts']

            # Define a list of colors for the plots
            # Add more colors if needed
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # Convert the numpy array to a DataFrame
            forecasts_df = pd.DataFrame(forecasts, columns=train.columns)

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train)), train[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts_df[column][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Training, Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts_df[column][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

        if selected_option == "Multivariate VARMAX":
            results = {}
            rmse_values = []

            for node_index_forecast in range(num_nodes):
                node_data_forecast = get_node_data_from_merged(
                    merged_data=merged_data, node_index=node_index_forecast)
                rmse, train, test, forecasts = multivariate_varmax_forecast(
                    df=node_data_forecast)
                results[node_index_forecast] = {
                    'rmse': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
                rmse_values.append(rmse)

            avg_rmse = np.mean(rmse_values)
            st.metric(f'Average RMSE', avg_rmse)

            node_index = st.slider(label="Node ID", min_value=0, max_value=max(
                num_nodes-1, 1), key='mul_varmax')
            node_data = get_node_data_from_merged(
                merged_data=merged_data, node_index=node_index)

            st.subheader("Node Data")
            st.dataframe(node_data)

            result = results[node_index]
            train = result['train']
            test = result['test']
            forecasts = result['forecasts']

            # Define a list of colors for the plots
            # Add more colors if needed
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # Convert the numpy array to a DataFrame
            forecasts_df = pd.DataFrame(forecasts, columns=train.columns)

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train)), train[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts_df[column][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Training, Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts_df[column][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

    if selected_forecast_option == "Multivariate K-hop":
        pass
