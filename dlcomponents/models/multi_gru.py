import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, History
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from dlcomponents.models.multi_preprocess import df_to_X_y3,preprocess_output,preprocess,plot_predictions
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import time
import plotly.express as px
import streamlit as st

# Suppress the deprecation warning for plt.show() usage
st.set_option('deprecation.showPyplotGlobalUse', False)

def multi_GRU(temp, target_columns):
    model = Sequential()
    model.add(GRU(64, activation='relu', input_shape=(7, temp.shape[1])))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(len(target_columns), activation='linear'))

    model.summary()

    cp = ModelCheckpoint('model.keras', save_best_only=True)
    history = History()
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    WINDOW_SIZE = 7
    X, y = df_to_X_y3(temp, WINDOW_SIZE, target_columns)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.75, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

    # Calculate mean and std for normalization
    mean_std_pairs = [(np.mean(X_train[:, :, i]), np.std(X_train[:, :, i])) for i in range(X_train.shape[2])]
    target_mean_std_pairs = [(np.mean(y_train[:, i]), np.std(y_train[:, i])) for i in range(y_train.shape[1])]

    # Preprocess the data
    preprocess(X_train, mean_std_pairs)
    preprocess(X_val, mean_std_pairs)
    preprocess(X_test, mean_std_pairs)
    preprocess_output(y_train, target_mean_std_pairs)
    preprocess_output(y_val, target_mean_std_pairs)
    preprocess_output(y_test, target_mean_std_pairs)


    epochs = 10
    progress_bar = st.progress(0)
    epoch_text = st.empty()
    loss_text = st.empty()
    val_loss_text = st.empty()

    for epoch in range(epochs):
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, callbacks=[cp], verbose=0)
        time.sleep(1)  # Simulate training time

        # Update progress bar and display epoch, loss, and validation loss
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        epoch_text.text(f"Epoch: {epoch + 1}")
        loss_text.text(f"Loss: {history.history['loss'][-1]:.4f}")
        val_loss_text.text(f"Val Loss: {history.history['val_loss'][-1]:.4f}")

    model = load_model('model.keras')

    # Predictions and Actuals table
    test_predictions = model.predict(X_test)
    # Create a DataFrame with interleaved prediction and actual columns
    df_results = pd.DataFrame()
    for i, col in enumerate(target_columns):
        df_results[f'{col} Predictions'] = test_predictions[:, i]
        df_results[f'{col} Actuals'] = y_test[:, i]
    st.write("Predicted vs Actual Values:")
    st.write(df_results)

    # Calculate and display additional metrics
    for i, col in enumerate(target_columns):
        mae = mean_absolute_error(y_test[:, i], test_predictions[:, i])
        r2 = r2_score(y_test[:, i], test_predictions[:, i])
        st.write(f"Mean Absolute Error ({col}): {mae:.4f}")
        st.write(f"R² Score ({col}): {r2:.4f}")

    # Plot predictions vs actuals using Plotly
    plot_predictions(model, X_test, y_test, target_columns)

