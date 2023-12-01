import time
from datetime import datetime

import numpy as np
import config


def to_timestamp(s):
    return np.int32(time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple()))


def calculate_log_close_mean(df, lag):
    """
    Calculates the log close mean for a given lag in-place.
    """
    df[f"log_close/mean_{lag}"] = np.log(
        df["close"]
        / np.roll(
            np.append(
                np.convolve(df["close"], np.ones(lag) / lag, mode="valid"),
                np.ones(lag - 1),
            ),
            lag - 1,
        )
    )


def calculate_log_return(df, lag):
    """
    Calculates the log return for a given lag in-place.
    """
    df[f"log_return_{lag}"] = np.log(df["close"] / np.roll(df["close"], lag))


def process_train_data(df):
    """
    Processes data for training, including setting flags and filtering by date.
    """
    valid_window = [to_timestamp("12/03/2021")]
    df["train_flg"] = np.where(df.index >= valid_window[0], 0, 1)
    oldest_use_window = [to_timestamp("12/01/2019")]
    return df[df.index >= oldest_use_window[0]]


def get_features(df, train=True):
    """
    Generates features for the given dataframe.
    """
    # TODO - see if skipping train_flg does anything
    # if train:
    #     df = process_train_data(df)

    # Calculate log close mean and log return for each lag and add as columns
    for lag in config.LAGS:
        calculate_log_close_mean(df, lag)
        calculate_log_return(df, lag)

        # Calculate mean close and mean log returns
        df[f"mean_close/mean_{lag}"] = np.mean(df[f"log_close/mean_{lag}"])
        df[f"mean_log_returns_{lag}"] = np.mean(df[f"log_return_{lag}"])

        # Additional calculations
        df[f"log_close/mean_{lag}-mean_close/mean_{lag}"] = (
            df[f"log_close/mean_{lag}"] - df[f"mean_close/mean_{lag}"]
        )
        df[f"log_return_{lag}-mean_log_returns_{lag}"] = (
            df[f"log_return_{lag}"] - df[f"mean_log_returns_{lag}"]
        )

    # TODO - check later if this is necessary
    # Additional processing for training data
    # if train:
    #     oldest_use_window = [to_timestamp("12/01/2019")]
    #     df = df[df["timestamp"] >= oldest_use_window[0]]

    return df
