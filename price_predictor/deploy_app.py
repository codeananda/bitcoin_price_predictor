import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from io import BytesIO
import plotly.express as px
import tensorflow as tf

from deploy_helpers import get_last_8_days_hourly_bitcoin_data

from helpers import scale_train_val, timeseries_to_keras_input, calculate_predictions

TIMESTEPS = 168
BATCH_SIZE = 100

title = "Simple Bitcoin Price Predictor"
st.set_page_config(
    page_title=title,
    page_icon="📈",
    layout="wide",
)

st.title(title)

left_col, right_col = st.columns(2)


def preprocess_data(df):
    price = df.price.values
    train_scaled, val_scaled = scale_train_val(
        price, np.zeros(10), scaler="log_and_range_0_1"
    )
    # Get data into form Keras needs
    X_train, X_val, y_train, y_val = timeseries_to_keras_input(
        train_scaled,
        val_scaled,
        input_seq_length=TIMESTEPS,
        output_seq_length=1,
        is_rnn=True,
        batch_size=BATCH_SIZE,
    )

    return X_train, X_val, y_train, y_val


with left_col:
    bitcoin = get_last_8_days_hourly_bitcoin_data()

    bitcoin

    fig = px.line(bitcoin, x="date", y="price")

    st.plotly_chart(fig)

with right_col:

    model_path = "../models/pretty-vortex-422/pretty-vortex-422-model-best.h5"
    model = tf.keras.models.load_model(model_path)

    # X_val, y_train, y_val are just placeholders
    # TO DO: Create Preprocessor, Plotter, Predictor etc. classes
    # or add a switch to functions like is_training to return relevant bits

    model_input_data, X_val, y_train, y_val = preprocess_data(bitcoin)
    # pred = model.predict(model_input_data)

    pred, _ = calculate_predictions(
        model,
        model_input_data,
        X_val,
        # y_train,
        # y_val,
        model_type="LSTM",
        batch_size=BATCH_SIZE,
    )

    st.write(f"In one hour, we predict the price will be {pred}")
