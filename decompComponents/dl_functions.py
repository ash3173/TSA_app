#File to preprocess, change the input dimensions and plot the output.

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

def df_to_X_y3(df, window_size, targets):
  df_as_np = df.to_numpy()
  X = []
  y = []
  target_indices = [df.columns.get_loc(target) for target in targets]
  
  for i in range(len(df_as_np) - window_size):
      row = [r for r in df_as_np[i:i + window_size]]
      X.append(row)
      label = [df_as_np[i + window_size][target_index] for target_index in target_indices]
      y.append(label)
  
  return np.array(X), np.array(y)


# Preprocessing function to normalize features
def preprocess(X, mean_std_pairs):
    for i, (mean, std) in enumerate(mean_std_pairs):
        X[:, :, i] = (X[:, :, i] - mean) / std
    return X

# Preprocessing function for outputs to normalize targets
def preprocess_output(y, mean_std_pairs):
    for i, (mean, std) in enumerate(mean_std_pairs):
        y[:, i] = (y[:, i] - mean) / std
    return y

def postprocess(predictions, mean_std_pairs):
    for i, (mean, std) in enumerate(mean_std_pairs):
        predictions[:, i] = (predictions[:, i] * std) + mean

    return predictions
    
# Function to plot predictions for multiple target variables
def plot_predictions(model, X, y, headers, start=0, end=100):
    predictions = model.predict(X)
    
    # Prepare a dictionary to store data for the DataFrame
    data = {}
    for i, header in enumerate(headers):
        data[f'{header} Predictions'] = predictions[:, i]
        data[f'{header} Actuals'] = y[:, i]
    
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    
    # Filter the data based on start and end
    df_filtered = df[start:end]
    
    # Generate column names for plotting
    columns_to_plot = []
    for header in headers:
        columns_to_plot.append(f'{header} Predictions')
        columns_to_plot.append(f'{header} Actuals')
    
    # Create an interactive plot using Plotly
    fig = px.line(df_filtered, y=columns_to_plot,
                  labels={'value': 'Values', 'variable': 'Predictions vs Actuals'},
                  title='Predictions vs Actuals')
    
    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)
    
    return df
