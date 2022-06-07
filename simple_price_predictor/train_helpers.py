import pandas as pd
from simple_price_predictor.constants import HISTORICAL_BITCOIN_CSV_FILEPATH
import os
import tensorflow as tf
import numpy as np


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
    array: np.ndarray, input_seq_length: int, output_seq_length: int, batch_size: int,
) -> tf.data.Dataset:
    """Return tf.data.Dataset that yeilds a tuple of input and output sequences
    of specified length. All batches are the same length.

    Parameters
    ----------
    array : np.ndarray
        2D input array
    input_seq_length : int
        Number of elements desired for input sequence
    output_seq_length : int
        Number of elements to forecast in output sequence
    batch_size : int
        Number of elements for each batch

    Returns
    -------
    tf.data.PrefetchDataset
        <PrefetchDataset shapes: ((batch_size, input_seq_length, 1),
            (batch_size, output_seq_length, 1)), types: (tf.int64, tf.int64)>

    Raises
    ------
    ValueError
        If array is not 2D

    """
    if len(array.shape) != 2:
        raise ValueError(
            f"`array` must be 2D array. Received:"
            f"{len(array.shape)}D array with array.shape={array.shape}"
        )

    total_seq_length = input_seq_length + output_seq_length

    ds = tf.data.Dataset.from_tensor_slices(array)

    ds = (
        ds.window(total_seq_length, shift=1, drop_remainder=True)
        .flat_map(lambda w: w.batch(total_seq_length, drop_remainder=True))
        .map(lambda w: (w[:-output_seq_length], w[-output_seq_length:]))
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds
