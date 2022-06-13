from simple_price_predictor.constants import HISTORICAL_BITCOIN_CSV_FILEPATH
from simple_price_predictor.train_helpers import (
    load_raw_bitcoin_df,
    make_tf_dataset,
)
import os
import tensorflow as tf
import numpy as np
import pytest


DATASET_SIZE = 1000000


@pytest.fixture
def input_array_2D():
    return np.arange(DATASET_SIZE).reshape(-1, 1)


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
    def test_array_greater_than_2D(self):
        array_3D = np.arange(DATASET_SIZE)
        array_3D = array_3D.reshape(-1, 10, 5)

        with pytest.raises(ValueError) as exec_info:
            make_tf_dataset(array_3D, 5, 5, 5)

    def test_array_less_than_2D(self):
        array_1D = np.arange(DATASET_SIZE)

        with pytest.raises(ValueError) as exec_info:
            make_tf_dataset(array_1D, 5, 5, 5)

    def test_zero_output_length(self, input_array_2D):
        input_seq_length = 200
        output_seq_length = 0
        batch_size = 20

        expected_feature_shape = (batch_size, input_seq_length, 1)
        expected_target_shape = (batch_size, output_seq_length, 1)

        ds = make_tf_dataset(
            input_array_2D, input_seq_length, output_seq_length, batch_size
        )

        for x in ds.take(1):
            feature, target = x

        assert expected_feature_shape == feature.shape
        assert expected_target_shape == target.shape

    def test_single_batch(self):
        pass
