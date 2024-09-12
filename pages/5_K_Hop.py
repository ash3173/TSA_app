import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import plotly.express as px
from khopcomponents.functions import create_sequences_for_prediction, aggregate_node_with_neighbors, k_hop_neighbors

# Set the title of the Streamlit app
st.title("Time Series Analysis using K-Hop")

# File uploader widgets in the sidebar for node and edge data files
node = st.sidebar.file_uploader("Upload a Node data file", type=["csv"])
edge = st.sidebar.file_uploader("Upload an Edge data file", type=["csv"])

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

if node_data is not None and edge_data is not None:
    # Sidebar parameters for node, feature, and k-hop value
    target_node = st.sidebar.number_input("Select Target Node", min_value=0, max_value=len(node_data["node"].unique()), value=0)
    target_feature = st.sidebar.number_input("Select Target Feature", min_value=0, max_value=len(node_data["feature"].unique())-1, value=0)
    k = st.sidebar.number_input("Select K-Hop Value", min_value=1, max_value=5, value=1)
    sequence_length = st.sidebar.slider("Select Sequence Length", min_value=3, max_value=30, value=7)

    # Aggregate node data with neighbors
    aggregated_series = aggregate_node_with_neighbors(node_data, edge_data, k, target_node, target_feature)

    # Create sequences for prediction
    X, y = create_sequences_for_prediction(aggregated_series, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 2))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 2))

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 2)))
    model.add(Dense(1))  # Output layer with 1 neuron for predicting the node's value
    model.compile(optimizer='adam', loss='mse')

    # Training the model with a progress bar
    epochs = st.sidebar.slider("Select Number of Epochs", min_value=10, max_value=500, value=200)
    progress_bar = st.progress(0)

    for epoch in range(epochs):
        model.fit(X_train, y_train, epochs=1, verbose=0, validation_split=0.1, batch_size=32)
        progress_bar.progress((epoch + 1) / epochs)

    # Evaluate model performance
    loss = model.evaluate(X_test, y_test)
    st.write(f"Test loss: {loss}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Plot actual vs. predicted values using Plotly Express
    st.subheader("Actual vs. Predicted Values")
    fig = px.line(
        x=np.arange(len(y_test)),
        y=[y_test.flatten(), y_pred.flatten()],
        labels={'x': 'Time Steps', 'value': 'Feature Value'},
        title="Actual vs. Predicted Values"
    )
    fig.update_traces(name='Actual', selector=dict(name='wide_variable_0'))
    fig.update_traces(name='Predicted', selector=dict(name='wide_variable_1'))
    st.plotly_chart(fig)
