import pandas as pd
from simple_price_predictor.constants import HISTORICAL_BITCOIN_CSV_FILEPATH
import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

import wandb
from wandb.keras import WandbCallback


def load_raw_bitcoin_df():
    """Load the downloads/price.csv dataset of historical BTC data from 2010-2021
    and return as a dataframe without an NaN values.

    Returns
    -------
    pd.DataFrame
        Dataframe of BTC close data from 2010-2021
    """
    bitcoin = pd.read_csv(
        HISTORICAL_BITCOIN_CSV_FILEPATH,
        index_col=0,
        parse_dates=True,
        names=["date", "price", "h", "l", "o"],
        usecols=["date", "price"],
        header=0,
    )
    bitcoin = bitcoin.dropna()
    return bitcoin


def make_tf_dataset(
    array: np.ndarray,
    input_seq_length: int = 200,
    output_seq_length: int = 1,
    batch_size: int = 20,
) -> tf.data.Dataset:
    """Return tf.data.Dataset that yeilds a tuple of input and output sequences
    of specified length. All batches are the same length.

    Can be used for train, val, and test data.

    Parameters
    ----------
    array : np.ndarray
        2D input array
    input_seq_length : int
        Number of elements desired for input sequence
    output_seq_length : int
        Number of elements to forecast in output sequence
    batch_size : int
        Number of input/output sequences to include in each batch

    Returns
    -------
    tf.data.PrefetchDataset
        <PrefetchDataset shapes: ((batch_size, input_seq_length, 1),
            (batch_size, output_seq_length, 1)), types: (tf.int64, tf.int64)>

    Raises
    ------
    ValueError
        - If array is not np.ndarray
        - If array is not 2D

    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f"`array` must be a numpy array. Received: {type(array)}")

    if len(array.shape) != 2:
        raise ValueError(
            f"`array` must be 2D array. Received: "
            f"{len(array.shape)}D array with shape={array.shape}"
        )

    num_possible_batches = (
        len(array) - input_seq_length - output_seq_length
    ) / batch_size

    if num_possible_batches < 1:
        raise ValueError(
            f"Cannot make a single batch with these inputs. "
            f"Either use more data or decrease `input_seq_length` or `batch_size`. "
            f" Quickest results come from decreasing `batch_size`."
        )

    if input_seq_length > len(array):
        raise ValueError(
            f"`input_seq_length` is bigger than len(array) so it's not possible "
            f"to create any batches of data."
        )

    total_seq_length = input_seq_length + output_seq_length

    ds = tf.data.Dataset.from_tensor_slices(array)

    ds = (
        ds.window(total_seq_length, shift=1, drop_remainder=True)
        .flat_map(lambda w: w.batch(total_seq_length, drop_remainder=True))
        .map(lambda w: (w[:input_seq_length], w[input_seq_length:]))
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds


def get_optimizer(optimizer="adam", learning_rate=1e-4):
    """Given an optimizer and a learning rate, return the optimizer
    object with the learning rate set.

    Parameters
    ----------
    optimizer : str, optional {'adam', 'rmsprop'}
        The Keras optimizer you would like to use (case insensitive input)
    learning_rate : float, optional, default 1e-4
        The learning rate

    Returns
    -------
    optimizer: tf.keras.optimizer object
        Optimizer object with the given learning rate
    """
    if optimizer.lower() == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer.lower() == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(
            f"Supported optimizers are: Adam and RMSprop. Received: {optimizer}"
        )
    return optimizer


def build_LSTM_training(
    optimizer="adam",
    learning_rate=1e-4,
    loss="mse",
    units=50,
    batch_size=9,
    timesteps=200,
    num_layers=2,
):
    """Build, compile and return an LSTM model of with the given params

    Parameters
    ----------
    optimizer : str, optional {'adam', 'rmsprop'}
        The Keras optimizer you would like to use (case insensitive input), by
        default 'adam'
    learning_rate : float, optional
        The learning rate, by default 1e-4
    loss : str, optional {all keras losses are accepted}
        Keras loss to use, by default 'mse'
    units : int, optional
        The number of nodes in each LSTM layer, by default 50
    batch_size : int, optional
        The number of sequences fed into the LSTM on each batch, by default 9
    timesteps : int, optional
        The length of each sequence fed into the model, by default 168
        (i.e., one week's worth of hourly data)
    num_layers : int, optional
        The total number of LSTMs to stack, by default 2

    Returns
    -------
    model
        Built and compiled LSTM model with the given params.
    """
    if not isinstance(num_layers, int):
        raise TypeError(f"`num_layers` must be an int but receieved {type(num_layers)}")
    if num_layers < 1:
        raise ValueError(
            f"You must pass at least 1 `num_layers`. Received: {num_layers}"
        )
    if units == 0:
        raise ValueError(f"`units` must be greater than 0. Received: {units}")
    ## BUILD LSTM
    # Add (num_layers - 1) layers that return sequences
    lstm_list = [
        LSTM(
            units,
            return_sequences=True,
            stateful=True,
            batch_input_shape=(batch_size, timesteps, 1),
        )
        for _ in range(num_layers - 1)
    ]
    # Add a final layer that does not return sequences
    lstm_list.append(
        LSTM(
            units,
            return_sequences=False,
            stateful=True,
            batch_input_shape=(batch_size, timesteps, 1),
        )
    )
    # Single node output layer
    lstm_list.append(Dense(1))
    model = Sequential(lstm_list)

    ## COMPILE LSTM
    optimizer_object = get_optimizer(optimizer=optimizer, learning_rate=learning_rate)
    model.compile(
        loss=loss, optimizer=optimizer_object, metrics=[RootMeanSquaredError()]
    )
    return model
