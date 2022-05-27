import pandas as pd
import numpy as np
import tensorflow as tf

from simple_price_predictor.train_helpers import load_bitcoin_df


bitcoin = load_bitcoin_df()

# Processing steps to take
# 1. Scale - scale_train_val - log_and_range_0_1 the best
# 2. Transform to Keras input - timeseries_to_keras_input

# Use tf.keras.utils.timeseries_dataset_from_array ?
# Helpful article https://www.tensorflow.org/tutorials/structured_data/time_series
