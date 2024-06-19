import streamlit as st
import pandas as pd
from dlcomponents.functions import get_node_data_from_merged,get_feature_with_dates
import matplotlib.pyplot as plt
from dlcomponents.models.single_lstm import single_LSTM
from dlcomponents.models.single_cnn import single_CNN
from dlcomponents.models.single_gru import single_GRU

# Set the title of the Streamlit app
st.title("Time Series Analysis using Deep Learning.")

# File uploader widgets in the sidebar for node and edge data files
node = st.sidebar.file_uploader("Upload a Node data file", type=["csv"])
edge = st.sidebar.file_uploader("Upload a Edge data file", type=["csv"])

# Dropdown menu for model selection
options = ["Select", "CNN", "LSTM", "GRU"]
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

    feature_indices = st.multiselect("Select feature indices for multivariate analysis", list(range(num_features)), default=[0])
    selected_node_features_data = selected_node_data.iloc[:, feature_indices]
    st.write(selected_node_features_data)

    if st.button("Train LSTM Model"):
        st.write("In progress.")
