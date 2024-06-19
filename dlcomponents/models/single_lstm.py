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
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import time

def single_LSTM(temp):
    model1 = Sequential()
    model1.add(InputLayer((5, 1)))
    model1.add(LSTM(64))
    model1.add(Dense(8, 'relu'))
    model1.add(Dense(1, 'linear'))

    cp1 = ModelCheckpoint('model1/', save_best_only=True)
    model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])

    WINDOW_SIZE = 5
    X1, y1 = df_to_X_y(temp, WINDOW_SIZE)

    X_temp, X_test1, y_temp, y_test1 = train_test_split(X1, y1, test_size=0.10, random_state=42)
    X_train1, X_val1, y_train1, y_val1 = train_test_split(X_temp, y_temp, test_size=1/9, random_state=42)


    epochs = 10
    progress_bar = st.progress(0)
    epoch_text = st.empty()
    for epoch in range(epochs):
        model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=1, callbacks=[cp1], verbose=0)
        time.sleep(1)  # Simulate training time
        progress = (epoch + 1) / epochs
        epoch_text.text(f"Epoch: {epoch + 1}")
        progress_bar.progress(progress)



    test_predictions = model1.predict(X_test1).flatten()
    test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test1})
    st.write(test_results)
    plt.plot(test_results['Test Predictions'][:50], label='Test Predictions')
    plt.plot(test_results['Actuals'][:50], label='Actuals')
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.title('Comparison of Test Predictions and Actuals')
    plt.legend()

    # Display the plot in Streamlit
    st.pyplot()