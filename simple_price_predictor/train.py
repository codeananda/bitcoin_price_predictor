import pandas as pd
import numpy as np
import tensorflow as tf
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.preprocessing import MinMaxScaler

from simple_price_predictor.train_helpers import load_raw_bitcoin_df


# Processing steps to take
# 1. Split
# 2. Scale - scale_train_val - log_and_range_0_1 the best
# 3. Transform to Keras input - timeseries_to_keras_input

# Use tf.keras.utils.timeseries_dataset_from_array ?
# Helpful article https://www.tensorflow.org/tutorials/structured_data/time_series


def main():
    bitcoin = load_raw_bitcoin_df()

    # In total we have: ~70% training, 20% val, 10% test
    train, test = temporal_train_test_split(bitcoin, train_size=0.9)
    train, val = temporal_train_test_split(train, train_size=0.77)

    train = np.log(train)
    val = np.log(val)
    test = np.log(test)

    min_max = MinMaxScaler()

    train = min_max.fit_transform(train)
    val = min_max.transform(val)
    test = min_max.transform(test)
