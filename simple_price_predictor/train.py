import pandas as pd
import numpy as np
import tensorflow as tf
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.preprocessing import MinMaxScaler

from simple_price_predictor.train_helpers import (
    load_raw_bitcoin_df,
    make_tf_dataset,
    build_LSTM_training,
)

from tensorflow.keras.callbacks import EarlyStopping


# Processing steps
# 1. Split
# 2. Scale - scale_train_val - log_and_range_0_1 the best
# 3. Transform to Keras input - timeseries_to_keras_input

# Model fit + building steps
# 1. Build training model
# 2. Get callbacks
# 3. Fit model

BATCH_SIZE = 50
OUTPUT_SEQ_LENGTH = 1
INPUT_SEQ_LENGTH = 200


def main():
    bitcoin = load_raw_bitcoin_df()

    # In total we have: ~70% training, 20% val, 10% test
    train, test = temporal_train_test_split(bitcoin, train_size=0.05)
    train, val = temporal_train_test_split(train, train_size=0.77)

    train = np.log(train)
    val = np.log(val)
    test = np.log(test)

    min_max = MinMaxScaler()

    train = min_max.fit_transform(train)
    val = min_max.transform(val)
    test = min_max.transform(test)

    train_ds = make_tf_dataset(
        train,
        input_seq_length=INPUT_SEQ_LENGTH,
        output_seq_length=OUTPUT_SEQ_LENGTH,
        batch_size=BATCH_SIZE,
    )
    val_ds = make_tf_dataset(
        val,
        input_seq_length=INPUT_SEQ_LENGTH,
        output_seq_length=OUTPUT_SEQ_LENGTH,
        batch_size=BATCH_SIZE,
    )
    test_ds = make_tf_dataset(
        test,
        input_seq_length=INPUT_SEQ_LENGTH,
        output_seq_length=OUTPUT_SEQ_LENGTH,
        batch_size=1,
    )

    model = build_LSTM_training(batch_size=BATCH_SIZE, timesteps=INPUT_SEQ_LENGTH)

    early_stop_cb = EarlyStopping(patience=10, restore_best_weights=True, baseline=None)

    callbacks = [early_stop_cb]

    history = model.fit(
        train_ds,
        epochs=10,
        shuffle=False,
        validation_data=val_ds,
        callbacks=callbacks,
        batch_size=BATCH_SIZE,
    )

    model.save("models/basic_LSTM.h5")


if __name__ == "__main__":
    main()
