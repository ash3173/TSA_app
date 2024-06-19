import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from arch import arch_model

def get_node_data_from_merged(merged_data, node_index):
    filtered = merged_data[merged_data["node"] == node_index].copy()
    filtered.drop(["node"], axis=1, inplace=True)
    unflattened = unflatten_dataframe(filtered)
    return unflattened

def unflatten_dataframe(df_flat):
    df = df_flat.pivot(index='timestamp', columns='feature', values='value')
    df.reset_index(drop=True, inplace=True)
    df.columns.name = None
    return df

def run_arch_model(node_data, edge_data):
    merged_data = node_data
    num_nodes = len(node_data["node"].unique())
    a = get_node_data_from_merged(merged_data, 0)
    
    # Perform rolling forecasts and plot
    num_features = a.shape[1]
    fig, axs = plt.subplots(num_features, 1, figsize=(14, 8 * num_features), sharex=True)

    for idx, feature in enumerate(a.columns):
        rolling_predictions = []

        # Rolling forecast loop starting from the last 100 values
        test_size = len(a)
        for i in range(test_size - 100, len(a)):
            train = a[feature].iloc[:i]

            try:
                model = arch_model(train, vol='ARCH', p=5)  # Example ARCH model, adjust parameters as needed
                model_fit = model.fit(disp='off', last_obs=i)  # Specify last_obs to avoid empty array error
            except ValueError as e:
                st.error(f"ValueError during model fitting: {e}")
                st.stop()

            pred = model_fit.forecast(horizon=1)
            rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

        rolling_predictions = pd.Series(rolling_predictions, index=a.index[test_size-100:])

        # Plot the last 100 actual values and the rolling forecasted values
        axs[idx].plot(a.index[-100:], a[feature].iloc[-100:], label='Actual')
        axs[idx].plot(rolling_predictions.index, rolling_predictions, label='Rolling Forecast', linestyle='--')
        axs[idx].set_title(f'Actual vs Rolling Forecast for Feature {feature}', size=20)
        axs[idx].set_xlabel('Time', size=16)
        axs[idx].set_ylabel('Value', size=16)
        axs[idx].legend()

    plt.tight_layout()
    st.pyplot(fig)  # Use Streamlit to display the figure
