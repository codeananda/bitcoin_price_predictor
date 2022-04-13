import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from io import BytesIO
import plotly.express as px
import tensorflow as tf

from deploy_helpers import get_last_8_days_hourly_bitcoin_data

from .price_predictor.helpers import scale_train_val

TIMESTEPS = 168
BATCH_SIZE = 100


def preprocess_data(df):
    price = df.price.values
    train_scaled, val_scaled = scale_train_val(
        price, np.zeros(10), scaler="log_and_range_0_1"
    )
    # Get data into form Keras needs
    X_train, _, _, _ = timeseries_to_keras_input(
        train_scaled,
        val_scaled,
        input_seq_length=TIMESTEPS,
        output_seq_length=1,
        is_rnn=True,
        batch_size=BATCH_SIZE,
    )

    return X_train


st.title("Predict the Price of Bitcoin in 1 Hour's Time")

if st.button("Get Recent Bitcoin Data"):
    bitcoin = get_last_8_days_hourly_bitcoin_data()

    bitcoin

    fig = px.line(bitcoin, x="date", y="price")

    st.plotly_chart(fig)

if st.button("Make Prediction for Next Hour"):
    model_path = "../models/pretty-vortex-422/pretty-vortex-422-model-best.h5"
    model = tf.keras.load_model(model_path)

    model_input_data = preprocess_data(bitcoin)
    pred = model.predict(model_input_data)

    st.write(f"In one hour, we predict the price will be {pred}")
