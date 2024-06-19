import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from dlcomponents.models.preprocess import df_to_X_y
from dlcomponents.models.multi_preprocess import df_to_X_y3
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import time
import plotly.express as px

def multi_LSTM(temp,sub1, sub2):
    model5 = Sequential()
    model5.add(InputLayer((7, temp.shape[1])))
    model5.add(LSTM(64))
    model5.add(Dense(8, 'relu'))
    model5.add(Dense(2, 'linear'))

    model5.summary()

    cp5 = ModelCheckpoint('model5/', save_best_only=True)
    model5.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])


    WINDOW_SIZE = 7
    X, y = df_to_X_y3(temp, WINDOW_SIZE,sub1, sub2)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.75, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

    p_training_mean = np.mean(X_train[:, :, 0])
    p_training_std = np.std(X_train[:, :, 0])

    temp_training_mean = np.mean(X_train[:, :, 1])
    temp_training_std = np.std(X_train[:, :, 1])

    def preprocess(X):
        X[:, :, 0] = (X[:, :, 0] - p_training_mean) / p_training_std
        X[:, :, 1] = (X[:, :, 1] - temp_training_mean) / temp_training_std 
        return X

    def preprocess_output(y):
        y[:, 0] = (y[:, 0] - p_training_mean) / p_training_std
        y[:, 1] = (y[:, 1] - temp_training_mean) / temp_training_std
        return y

    preprocess(X_train)
    preprocess(X_val)
    preprocess(X_test)

    preprocess_output(y_train)
    preprocess_output(y_val)
    preprocess_output(y_test)

    epochs = 10
    progress_bar = st.progress(0)
    epoch_text = st.empty()
    for epoch in range(epochs):
        model5.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, callbacks=[cp5], verbose=0)
        time.sleep(1)  # Simulate training time
        progress = (epoch + 1) / epochs
        epoch_text.text(f"Epoch: {epoch + 1}")
        progress_bar.progress(progress)

    def plot_predictions2(model, X, y, headers, start=0, end=100):
        predictions = model.predict(X)
        p_preds, temp_preds = predictions[:, 0], predictions[:, 1]
        p_actuals, temp_actuals = y[:, 0], y[:, 1]
        
        df = pd.DataFrame(data={
            f'{headers[1]} Predictions': temp_preds,
            f'{headers[1]} Actuals': temp_actuals,
            f'{headers[0]} Predictions': p_preds,
            f'{headers[0]} Actuals': p_actuals
        })

        # Filter the data based on start and end
        df_filtered = df[start:end]

        # Create an interactive plot using Plotly
        fig = px.line(df_filtered, y=[f'{headers[1]} Predictions', f'{headers[1]} Actuals', f'{headers[0]} Predictions', f'{headers[0]} Actuals'],
                    labels={'value': 'Values', 'variable': 'Predictions vs Actuals'},
                    title='Predictions vs Actuals')

        # Display the interactive plot in Streamlit
        st.plotly_chart(fig)

        return df
    plot_predictions2(model5, X_test, y_test, headers=[sub1, sub2])