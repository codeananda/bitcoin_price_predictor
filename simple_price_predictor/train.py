import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.preprocessing import MinMaxScaler
import argparse
from pathlib import Path

from simple_price_predictor.train_helpers import (
    load_raw_bitcoin_df,
    make_tf_dataset,
    build_LSTM_training,
)

from simple_price_predictor.constants import (
    OUTPUT_SEQ_LENGTH,
    INPUT_SEQ_LENGTH,
    BATCH_SIZE_TRAINING,
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


def train_model(output_model_name):
    if not output_model_name.endswith(".h5"):
        raise ValueError(
            f"You must pass a model name with a .h5 extension at the end. "
            f"Received: {output_model_name}"
        )

    output_dir = Path("models")
    if not output_dir.exists():
        output_dir.mkdir()

    # Remove .h5 at the end
    output_model_name = output_model_name[:-3]
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

    with open(output_dir / f"{output_model_name}_scaler.pkl", "wb") as f:
        pickle.dump(min_max, f)

    train_ds = make_tf_dataset(
        train,
        input_seq_length=INPUT_SEQ_LENGTH,
        output_seq_length=OUTPUT_SEQ_LENGTH,
        batch_size=BATCH_SIZE_TRAINING,
    )
    val_ds = make_tf_dataset(
        val,
        input_seq_length=INPUT_SEQ_LENGTH,
        output_seq_length=OUTPUT_SEQ_LENGTH,
        batch_size=BATCH_SIZE_TRAINING,
    )
    test_ds = make_tf_dataset(
        test,
        input_seq_length=INPUT_SEQ_LENGTH,
        output_seq_length=OUTPUT_SEQ_LENGTH,
        batch_size=1,
    )

    model = build_LSTM_training(
        batch_size=BATCH_SIZE_TRAINING, timesteps=INPUT_SEQ_LENGTH
    )

    early_stop_cb = EarlyStopping(patience=10, restore_best_weights=True, baseline=None)

    callbacks = [early_stop_cb]

    history = model.fit(
        train_ds,
        epochs=10,
        shuffle=False,
        validation_data=val_ds,
        callbacks=callbacks,
        batch_size=BATCH_SIZE_TRAINING,
    )

    model.save(output_dir / f"{output_model_name}")
    model.save_weights(output_dir / f"{output_model_name}_weights.h5")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        "--output-model",
        required=True,
        help="path to output model file (.h5 extension)",
    )
    args = vars(ap.parse_args())

    train_model(args["output_model"])
