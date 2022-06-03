from simple_price_predictor.constants import HISTORICAL_BITCOIN_CSV_FILEPATH
from simple_price_predictor.train_helpers import load_raw_bitcoin_df
import os


class TestLoadRawBitcoinDF:
    def test_historical_bitcoin_csv_filepath_exists(self):
        assert HISTORICAL_BITCOIN_CSV_FILEPATH.exists()

    def test_load_raw_bitcoin_df(self):
        load_raw_bitcoin_df()

    def test_raw_bitcoin_data(self):
        bitcoin = load_raw_bitcoin_df()
        assert bitcoin.price.isna().sum() == 0
        assert bitcoin.price.min() > 0
        assert len(bitcoin.columns) == 1
