from simple_price_predictor.constants import HISTORICAL_BITCOIN_CSV_FILEPATH
from simple_price_predictor.train_helpers import load_bitcoin_df
import os


def test_historical_bitcoin_csv_filepath_exists():
    assert HISTORICAL_BITCOIN_CSV_FILEPATH.exists()


def test_load_bitcoin_df():
    load_bitcoin_df()
