import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, History
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import time
import plotly.express as px
import streamlit as st

from decompComponents.dl_functions import df_to_X_y3, preprocess_output, preprocess, plot_predictions, postprocess

# Suppress the deprecation warning for plt.show() usage
st.set_option('deprecation.showPyplotGlobalUse', False)

def multi_CNN_GRU(temp, target_columns, key):
    # Assume temp is your input data
    model = Sequential()
    model.add(InputLayer((7, temp.shape[1])))  # Assuming window size of 7
    model.add(Conv1D(64, kernel_size=2, activation='relu'))
    model.add(GRU(64, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(len(target_columns), activation='linear'))

    model.summary()

    cp = ModelCheckpoint('model.keras', save_best_only=True)
    history = History()
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001),
                  metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    WINDOW_SIZE = 7
    X, y = df_to_X_y3(temp, WINDOW_SIZE, target_columns)  # Prepare X, y for training

    train_size = int(len(X) * 0.7)  # Train on first 70% of the data
    val_size = int(len(X) * 0.2)  # Validate on next 15% of the data
    test_size = len(X) - train_size - val_size  # Test on final 15% of the data
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[-test_size:], y[-test_size:]

    # Calculate mean and std for normalization
    mean_std_pairs = [(np.mean(X_train[:, :, i]), np.std(X_train[:, :, i])) for i in range(X_train.shape[2])]
    target_mean_std_pairs = [(np.mean(y_train[:, i]), np.std(y_train[:, i])) for i in range(y_train.shape[1])]

    # Preprocess the data
    preprocess(X_train, mean_std_pairs)
    preprocess(X_test, mean_std_pairs)
    preprocess_output(y_train, target_mean_std_pairs)
    preprocess_output(y_test, target_mean_std_pairs)

    epochs = 30
    progress_bar = st.progress(0)
    epoch_text = st.empty()
    loss_text = st.empty()
    val_loss_text = st.empty()

    for epoch in range(epochs):
        history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=1, callbacks=[cp], verbose=0)
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
    
    # Postprocess predictions to denormalize
    test_predictions = postprocess(test_predictions, target_mean_std_pairs)
    y_test = postprocess(y_test, target_mean_std_pairs)

    test_predictions_dataframe = pd.DataFrame(
        test_predictions,index=temp.index[-len(test_predictions):] , columns=target_columns)

    df_results = pd.DataFrame()
    for i, col in enumerate(target_columns):
        df_results[f'{col} Predictions'] = test_predictions[:, i]
        df_results[f'{col} Actuals'] = y_test[:, i]

    fig = px.line(df_results, x=df_results.index, y=[f'{col} Predictions' for col in target_columns] + [
                  f'{col} Actuals' for col in target_columns], title='Predictions vs Actuals')
    st.plotly_chart(fig)

    return test_predictions_dataframe