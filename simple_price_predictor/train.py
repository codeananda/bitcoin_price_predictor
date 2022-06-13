import pandas as pd
import numpy as np
import tensorflow as tf
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.preprocessing import MinMaxScaler

from simple_price_predictor.train_helpers import load_raw_bitcoin_df, make_tf_dataset


# Processing steps
# 1. Split
# 2. Scale - scale_train_val - log_and_range_0_1 the best
# 3. Transform to Keras input - timeseries_to_keras_input

# Model fit + building steps
# 1. Build training model
# 2. Get callbacks
# 3. Fit model


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

    tf_dataset_params = dict()

    train_ds = make_tf_dataset(
        train, input_seq_length=200, output_seq_length=1, batch_size=20
    )
    val_ds = make_tf_dataset(
        val, input_seq_length=200, output_seq_length=1, batch_size=20
    )
    test_ds = make_tf_dataset(
        test, input_seq_length=200, output_seq_length=1, batch_size=1
    )

    # model = build_LSTM_training()

    # callbacks = get_callbacks()

    # history = model.fit()


if __name__ == "__main__":
    main()
