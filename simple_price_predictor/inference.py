import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import argparse
from pathlib import Path

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


def check_all_needed_files_exist(input_model_path):
    """Checks that the files needed to run inference are at their expected
    locations. Raises FileNotFoundErrors if they are not and suggests running
    train.py with the appropriate args to make it work.

    Parameters
    ----------
    input_model_path : str of form path/to/your/model.h5

    Raises
    ----------
    ValueError - if input_model_path does not end with .h5
    FileNotFoundError - if the files needed for inference do not exist.
    """
    if not input_model_path.endswith(".h5"):
        raise ValueError(
            f"You must pass a model path with a .h5 extension at the end. "
            f"Received: {input_model_path}"
        )
    # Turn path/to/model.h5 into model
    model_name = Path(input_model_path).parts[-1]
    model_name = model_name[:-3]

    if not Path(input_model_path).exists():
        raise FileNotFoundError(
            f"Cannot find {input_model_path}. "
            f"Have you remembered to run train.py with `--output-model "
            f"{model_name}` first?"
        )

    # Drop .h5
    input_model_path = input_model_path[:-3]

    scaler_path = f"{input_model_path}_scaler.pkl"
    if not Path(scaler_path).exists():
        raise FileNotFoundError(
            f"Cannot find {scaler_path}. "
            f"Have you remembered to run train.py with `--output-model "
            f"{model_name}` first?"
        )

    model_weights_path = f"{input_model_path}_weights.h5"
    if not Path(model_weights_path).exists():
        raise FileNotFoundError(
            f"Cannot find {model_weights_path}. "
            f"Have you remembered to run train.py with `--output-model "
            f"{model_name}` first?"
        )


def make_predictions(input_model_path):
    check_all_needed_files_exist(input_model_path)

    num_days = 9
    bitcoin = get_last_num_days_hourly_bitcoin_data(num_days)
    bitcoin = bitcoin.price.values.reshape(-1, 1)

    bitcoin = np.log(bitcoin)

    with open(f"{input_model_path}_scaler.pkl", "rb") as f:
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
    inference_model.load_weights(f"{input_model_path}_weights.h5")

    preds = inference_model.predict(bitcoin_ds)
    preds = pd.Series(preds.reshape(-1,), name="preds")

    return preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input-model", required=True, help="path to trained model file (.h5)",
    )
    ap.add_argument(
        "-o",
        "--output-data",
        required=True,
        help="path to file to store predictions (.txt or .csv)",
    )
    args = vars(ap.parse_args())

    preds = make_predictions(args["input_model"])

    y_pred = pd.Series(preds.reshape(-1,), name="preds")
    y_pred.to_csv(f"{args['output_data']}", header=False, index=False)
