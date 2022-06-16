import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from simple_price_predictor.inference_helpers import (
    get_last_num_days_hourly_bitcoin_data,
)


def main():

    num_days = 15
    bitcoin = get_last_num_days_hourly_bitcoin_data(num_days)

    with open("models/basic_scaler.pkl", "rb") as f:
        min_max = pickle.load(f)

    return 1


if __name__ == "__main__":
    main()
