from simple_price_predictor.constants import HISTORICAL_BITCOIN_CSV_FILEPATH
import os


def test_historical_bitcoin_csv_filepath_exists():
    assert HISTORICAL_BITCOIN_CSV_FILEPATH.exists()
