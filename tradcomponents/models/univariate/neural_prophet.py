import pandas as pd
from neuralprophet import NeuralProphet
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import streamlit as st

def get_node_data_from_merged(merged_data, node_index):
    filtered = merged_data[merged_data["node"] == node_index]
    filtered.drop(["node"], axis=1, inplace=True)
    unflattened = unflatten_dataframe(filtered)
    return unflattened

def unflatten_dataframe(df_flat):
    df = df_flat.pivot(index='timestamp', columns='feature', values='value')
    
    # Convert index to date format
    start_date = pd.to_datetime('2004-01-01')  # Adjust the start date as per your data
    df.index = start_date + pd.to_timedelta(df.index, unit='D')
    
    # Add new column 'ds' with timestamps
    df['ds'] = df.index
    
    df.reset_index(drop=True, inplace=True)
    df.columns.name = None
    return df

def run_neural_prophet_model(node_data, edge_data, feature_index):
    # Example usage
    merged_data = node_data
    num_nodes = len(node_data["node"].unique())

    a = merged_data[merged_data["node"] == 0].copy()  # Make a copy to avoid SettingWithCopyWarning
    a.drop(["node"], axis=1, inplace=True)

    a = unflatten_dataframe(a)

    # Convert feature_index to integer to match the format in the DataFrame
    feature_index = int(feature_index)

    # Check if the selected feature_index exists in the DataFrame columns
    if feature_index not in a.columns:
        st.error(f"Feature index {feature_index} not found in the data.")
        return

    # Create new DataFrame df with 'ds' and selected feature renamed to 'y'
    df = a[['ds', feature_index]].rename(columns={feature_index: 'y'})

    # Split data into training (80%) and test (20%)
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

    # Initialize NeuralProphet model with epochs=5
    model = NeuralProphet(epochs=5)

    # Fit the model on training data
    metrics = model.fit(train_df, freq='D')

    # Make predictions on test data
    predictions = model.predict(test_df)

    # Plot actual vs predicted for test data using Plotly
    fig = px.line(test_df, x='ds', y='y', labels={'y': f'Value for Feature {feature_index}'}, title=f'Actual vs Predicted (Test Data) for Feature {feature_index}')
    fig.add_scatter(x=test_df['ds'], y=predictions['yhat1'], mode='lines', name='Predicted', line=dict(dash='dash'))

    # Display the plot in Streamlit
    st.plotly_chart(fig)
