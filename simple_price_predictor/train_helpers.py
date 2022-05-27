import pandas as pd
from simple_price_predictor.constants import HISTORICAL_BITCOIN_CSV_FILEPATH
import os


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
    test_raw_bitcoin_data(bitcoin)
    return bitcoin


def test_raw_bitcoin_data(bitcoin):
    """Integration test for loading raw bitcoin data from disk."""
    assert bitcoin.price.isna().sum() == 0
    assert bitcoin.price.min() > 0
    assert len(bitcoin.columns) == 1
