import streamlit as st
import pandas as pd
from dlcomponents.functions import get_node_data_from_merged,get_feature_with_dates
import matplotlib.pyplot as plt
from dlcomponents.models.single_lstm import single_LSTM
from dlcomponents.models.single_cnn import single_CNN
from dlcomponents.models.single_gru import single_GRU
from dlcomponents.models.multi_lstm import multi_LSTM
from dlcomponents.models.multi_cnn import multi_CNN
from dlcomponents.models.multi_gru import multi_GRU
from dlcomponents.models.multi_cnn_gru import multi_CNN_GRU
from dlcomponents.models.multi_cnn_lstm import multi_CNN_LSTM

# Set the title of the Streamlit app
st.title("Time Series Analysis using Deep Learning.")

# File uploader widgets in the sidebar for node and edge data files
node = st.sidebar.file_uploader("Upload a Node data file", type=["csv"])
edge = st.sidebar.file_uploader("Upload a Edge data file", type=["csv"])

# Dropdown menu for model selection
options = ["Select", "CNN", "LSTM", "GRU", "CNN-GRU","CNN-LSTM"]
model_option = st.sidebar.selectbox("Choose a model", options)

analysis_type = st.sidebar.selectbox("Choose analysis type", ["Select", "Single Variate", "Multivariate"])

# Function to load the CSV data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Load the node and edge data
node_data = load_data(node)
edge_data = load_data(edge)

# Display the uploaded data
if node_data is not None:
    st.subheader("Node Data")
    st.write(node_data)

if edge_data is not None:
    st.subheader("Edge Data")
    st.write(edge_data)

# Model selection and corresponding message
if model_option == "Select":
    st.write("Upload the node and edge data.")
elif model_option == "LSTM" and analysis_type == "Single Variate":
    st.write("LSTM model selected for single-variate analysis.")
    num_nodes = len(node_data["node"].unique())
    num_features = len(node_data["feature"].unique())
    node_index = st.slider("Select the node index", 0, num_nodes - 1)

    temp = node_data
    st.write("merged data for node", node_index)
    selected_node_data = get_node_data_from_merged(merged_data=temp, node_index=node_index)
    st.write(selected_node_data)
    st.line_chart(selected_node_data)

    feature_index = st.slider("Select the feature index", 0, num_features - 1)
    selected_node_feature_data = get_feature_with_dates(df=selected_node_data, feature_index=feature_index)   
    
    temp=selected_node_data[feature_index]
    st.write(temp)
    if st.button("Train LSTM Model"):
        single_LSTM(temp)

elif model_option == "CNN" and analysis_type == "Single Variate":
    st.write("CNN model selected for single-variate analysis.")
    num_nodes = len(node_data["node"].unique())
    num_features = len(node_data["feature"].unique())
    node_index = st.slider("Select the node index", 0, num_nodes - 1)

    temp = node_data
    st.write("merged data for node", node_index)
    selected_node_data = get_node_data_from_merged(merged_data=temp, node_index=node_index)
    st.write(selected_node_data)
    st.line_chart(selected_node_data)

    feature_index = st.slider("Select the feature index", 0, num_features - 1)
    selected_node_feature_data = get_feature_with_dates(df=selected_node_data, feature_index=feature_index)   
    
    temp=selected_node_data[feature_index]
    st.write(temp)
    if st.button("Train CNN Model"):
        single_CNN(temp)

elif model_option == "GRU" and analysis_type == "Single Variate":
    st.write("GRU model selected for single-variate analysis.")
    num_nodes = len(node_data["node"].unique())
    num_features = len(node_data["feature"].unique())
    node_index = st.slider("Select the node index", 0, num_nodes - 1)

    temp = node_data
    st.write("merged data for node", node_index)
    selected_node_data = get_node_data_from_merged(merged_data=temp, node_index=node_index)
    st.write(selected_node_data)
    st.line_chart(selected_node_data)

    feature_index = st.slider("Select the feature index", 0, num_features - 1)
    selected_node_feature_data = get_feature_with_dates(df=selected_node_data, feature_index=feature_index)   
    
    temp=selected_node_data[feature_index]
    st.write(temp)
    if st.button("Train GRU Model"):
        single_GRU(temp)
elif model_option == "LSTM" and analysis_type == "Multivariate":
    st.write("LSTM model selected for multi-variate analysis.")
    num_nodes = len(node_data["node"].unique())
    num_features = len(node_data["feature"].unique())
    node_index = st.slider("Select the node index", 0, num_nodes - 1)

    temp = node_data
    st.write("Merged data for node", node_index)
    selected_node_data = get_node_data_from_merged(merged_data=temp, node_index=node_index)
    st.write(selected_node_data)
    st.line_chart(selected_node_data)

    feature_headers = selected_node_data.columns.tolist()
    selected_feature_headers = st.multiselect("Select feature headers for multivariate analysis", feature_headers, default=[feature_headers[0]])

    selected_node_features_data = selected_node_data[selected_feature_headers]
    st.write(selected_node_features_data)

    forecast_headers = st.multiselect("Select feature headers for forecasting", selected_feature_headers, default=selected_feature_headers[:2])

    if not forecast_headers:
        st.warning("Please select at least one feature header for forecasting.")
    else:
        selected_forecast_features_data = selected_node_features_data[forecast_headers]
        st.write("Selected features for forecasting:")
        st.write(selected_forecast_features_data)
        if st.button("Train LSTM Model"):
            multi_LSTM(selected_node_features_data,forecast_headers)

elif model_option == "CNN" and analysis_type == "Multivariate":
    st.write("CNN model selected for multi-variate analysis.")
    num_nodes = len(node_data["node"].unique())
    num_features = len(node_data["feature"].unique())
    node_index = st.slider("Select the node index", 0, num_nodes - 1)

    temp = node_data
    st.write("Merged data for node", node_index)
    selected_node_data = get_node_data_from_merged(merged_data=temp, node_index=node_index)
    st.write(selected_node_data)
    st.line_chart(selected_node_data)

    feature_headers = selected_node_data.columns.tolist()
    selected_feature_headers = st.multiselect("Select feature headers for multivariate analysis", feature_headers, default=[feature_headers[0]])

    selected_node_features_data = selected_node_data[selected_feature_headers]
    st.write(selected_node_features_data)

    forecast_headers = st.multiselect("Select feature headers for forecasting", selected_feature_headers, default=selected_feature_headers[:2])

    if not forecast_headers:
        st.warning("Please select at least one feature header for forecasting.")
    else:
        selected_forecast_features_data = selected_node_features_data[forecast_headers]
        st.write("Selected features for forecasting:")
        st.write(selected_forecast_features_data)
        
        if st.button("Train CNN Model"):
            multi_CNN(selected_node_features_data, forecast_headers)
elif model_option == "GRU" and analysis_type == "Multivariate":
    st.write("GRU model selected for multi-variate analysis.")
    num_nodes = len(node_data["node"].unique())
    num_features = len(node_data["feature"].unique())
    node_index = st.slider("Select the node index", 0, num_nodes - 1)

    temp = node_data
    st.write("Merged data for node", node_index)
    selected_node_data = get_node_data_from_merged(merged_data=temp, node_index=node_index)
    st.write(selected_node_data)
    st.line_chart(selected_node_data)

    feature_headers = selected_node_data.columns.tolist()
    selected_feature_headers = st.multiselect("Select feature headers for multivariate analysis", feature_headers, default=[feature_headers[0]])

    selected_node_features_data = selected_node_data[selected_feature_headers]
    st.write(selected_node_features_data)

    forecast_headers = st.multiselect("Select feature headers for forecasting", selected_feature_headers, default=selected_feature_headers[:2])

    if not forecast_headers:
        st.warning("Please select at least one feature header for forecasting.")
    else:
        selected_forecast_features_data = selected_node_features_data[forecast_headers]
        st.write("Selected features for forecasting:")
        st.write(selected_forecast_features_data)
        
        if st.button("Train GRU Model"):
            multi_GRU(selected_node_features_data, forecast_headers)
elif model_option == "CNN-GRU" and analysis_type == "Multivariate":
    st.write("CNN-GRU model selected for multi-variate analysis.")
    num_nodes = len(node_data["node"].unique())
    num_features = len(node_data["feature"].unique())
    node_index = st.slider("Select the node index", 0, num_nodes - 1)

    temp = node_data
    st.write("Merged data for node", node_index)
    selected_node_data = get_node_data_from_merged(merged_data=temp, node_index=node_index)
    st.write(selected_node_data)
    st.line_chart(selected_node_data)

    feature_headers = selected_node_data.columns.tolist()
    selected_feature_headers = st.multiselect("Select feature headers for multivariate analysis", feature_headers, default=[feature_headers[0]])

    selected_node_features_data = selected_node_data[selected_feature_headers]
    st.write(selected_node_features_data)

    forecast_headers = st.multiselect("Select feature headers for forecasting", selected_feature_headers, default=selected_feature_headers[:2])

    # Ensure only two headers are selected
    if not forecast_headers:
        st.warning("Please select at least one feature header for forecasting.")
    else:
        selected_forecast_features_data = selected_node_features_data[forecast_headers]
        st.write("Selected features for forecasting:")
        st.write(selected_forecast_features_data)

        if st.button("Train CNN-GRU Model"):
            multi_CNN_GRU(selected_node_features_data, forecast_headers)
elif model_option == "CNN-LSTM" and analysis_type == "Multivariate":
    st.write("CNN-LSTM model selected for multi-variate analysis.")
    num_nodes = len(node_data["node"].unique())
    num_features = len(node_data["feature"].unique())
    node_index = st.slider("Select the node index", 0, num_nodes - 1)

    temp = node_data
    st.write("Merged data for node", node_index)
    selected_node_data = get_node_data_from_merged(merged_data=temp, node_index=node_index)
    st.write(selected_node_data)
    st.line_chart(selected_node_data)

    feature_headers = selected_node_data.columns.tolist()
    selected_feature_headers = st.multiselect("Select feature headers for multivariate analysis", feature_headers, default=[feature_headers[0]])

    selected_node_features_data = selected_node_data[selected_feature_headers]
    st.write(selected_node_features_data)

    # New code to select two feature headers for forecasting
    forecast_headers = st.multiselect("Select feature headers for forecasting", selected_feature_headers, default=selected_feature_headers[:2])

    # Ensure only two headers are selected
    if not forecast_headers:
        st.warning("Please select at least one feature header for forecasting.")
    else:
        selected_forecast_features_data = selected_node_features_data[forecast_headers]
        st.write("Selected features for forecasting:")
        st.write(selected_forecast_features_data)

        if st.button("Train CNN-LSTM Model"):
            # Call your multi_CNN_LSTM function with selected features
            multi_CNN_LSTM(selected_forecast_features_data, forecast_headers)
