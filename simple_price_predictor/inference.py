import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from simple_price_predictor.constants import (
    OUTPUT_SEQ_LENGTH,
    INPUT_SEQ_LENGTH,
)

from simple_price_predictor.train_helpers import (
    make_tf_dataset,
    build_LSTM_training,
)

from simple_price_predictor.inference_helpers import (
    get_last_num_days_hourly_bitcoin_data,
)


def main():

    num_days = 9
    bitcoin = get_last_num_days_hourly_bitcoin_data(num_days)
    bitcoin = bitcoin.price.values.reshape(-1, 1)

    bitcoin = np.log(bitcoin)

    with open("models/basic_scaler.pkl", "rb") as f:
        min_max = pickle.load(f)

    bitcoin_preprocessed = min_max.transform(bitcoin)

    bitcoin_ds = make_tf_dataset(
        bitcoin_preprocessed,
        input_seq_length=INPUT_SEQ_LENGTH,
        output_seq_length=0,
        # Only feed in single batches for inference for flexibility
        batch_size=1,
    )

    inference_model = build_LSTM_training(batch_size=1, timesteps=INPUT_SEQ_LENGTH)
    inference_model.load_weights("models/basic_LSTM_weights.h5")

    preds = inference_model.predict(bitcoin_ds)

    print(preds)

    return preds


if __name__ == "__main__":
    main()
