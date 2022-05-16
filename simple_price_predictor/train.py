import pandas as pd
import numpy as np
import tensorflow as tf
from fastquant import get_crypto_data

# Data starts 08/2017 but good enough for now.
bitcoin = get_crypto_data("BTC/USDT", "2008-01-01", "2022-05-15")
bitcoin = bitcoin.close

# Processing steps to take
# 1. Scale - scale_train_val - log_and_range_0_1 the best
# 2. Transform to Keras input - timeseries_to_keras_input

# Use tf.keras.utils.timeseries_dataset_from_array ?
# Helpful article https://www.tensorflow.org/tutorials/structured_data/time_series
