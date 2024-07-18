import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, History
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError as MAE
from tensorflow.keras.optimizers import Adam
from dlcomponents.models.preprocess import df_to_X_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import streamlit as st
import time

# Suppress the deprecation warning for plt.show() usage
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.experimental_fragment
def single_LSTM(temp):
    model = Sequential([
        InputLayer((5, 1)),
        LSTM(64),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.summary()

    cp = ModelCheckpoint('model.keras', save_best_only=True)
    history = History()
    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(learning_rate=0.001),
                  metrics=[RootMeanSquaredError(), MeanAbsoluteError(), 'mae'])  # Adding MAE as a metric

    WINDOW_SIZE = 5
    X, y = df_to_X_y(temp, WINDOW_SIZE)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, random_state=42)

    epochs = 10
    progress_bar = st.progress(0)
    epoch_text = st.empty()
    loss_text = st.empty()
    val_loss_text = st.empty()

    for epoch in range(epochs):
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, callbacks=[cp], verbose=0)
        time.sleep(1)  # Simulate training time

        # Update progress bar, epoch info, loss and validation loss
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        epoch_text.text(f"Epoch: {epoch + 1}")
        loss_text.text(f"Loss: {history.history['loss'][-1]:.4f}")
        val_loss_text.text(f"Val Loss: {history.history['val_loss'][-1]:.4f}")

    model = load_model('model.keras')

    test_predictions = model.predict(X_test).flatten()
    test_results = pd.DataFrame(data={'Test Predictions': test_predictions, 'Actuals': y_test})
    st.write(test_results)

    # Calculate and display additional metrics: MAE and R^2 Score
    mae = tf.keras.metrics.MeanAbsoluteError()
    mae.update_state(y_test, test_predictions)
    mae_result = mae.result().numpy()
    st.write(f"Mean Absolute Error: {mae_result:.4f}")

    r2 = r2_score(y_test, test_predictions)
    st.write(f"R² Score: {r2:.4f}")

    plt.plot(test_results['Test Predictions'][:50], label='Test Predictions')
    plt.plot(test_results['Actuals'][:50], label='Actuals')
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.title('Comparison of Test Predictions and Actuals')
    plt.legend()

    # Display the plot in Streamlit
    st.pyplot()

    # Return the final test results DataFrame for potential further analysis or display
    return test_results
