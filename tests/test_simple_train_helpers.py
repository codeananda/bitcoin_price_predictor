from simple_price_predictor.constants import HISTORICAL_BITCOIN_CSV_FILEPATH
from simple_price_predictor.train_helpers import (
    load_raw_bitcoin_df,
    make_tf_dataset,
)
import os
import tensorflow as tf
import numpy as np
import pytest


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


class Test_make_tf_dataset:
    def test_runs(self):
        pass

    def test_array_greater_than_2D(self):
        pass

    def test_array_less_than_2D(self):
        array_1D = np.arange(100)

        with pytest.raises(ValueError) as exec_info:
            make_tf_dataset(array_1D, 5, 5, 5)
