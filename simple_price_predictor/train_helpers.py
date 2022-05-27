import pandas as pd
from constants import HISTORICAL_BITCOIN_CSV_FILEPATH
import os


def load_bitcoin_df():
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
