from simple_price_predictor.constants import HISTORICAL_BITCOIN_CSV_FILEPATH
from simple_price_predictor.train_helpers import (
    load_raw_bitcoin_df,
    make_tf_dataset,
)
import os
import tensorflow as tf
import numpy as np
import pytest
from pytest_cases import parametrize_with_cases


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

    def compare_feature_and_target_shapes(
        self, array, input_seq_length, output_seq_length, batch_size
    ):
        expected_feature_shape = (batch_size, input_seq_length, 1)
        expected_target_shape = (batch_size, output_seq_length, 1)

        ds = make_tf_dataset(array, input_seq_length, output_seq_length, batch_size)

        for x in ds.take(1):
            feature, target = x

        assert expected_feature_shape == feature.shape
        assert expected_target_shape == target.shape

    def test_output_seq_length_is_zero(self, input_array_2D):
        input_seq_length = 200
        output_seq_length = 0
        batch_size = 20

        self.compare_feature_and_target_shapes(
            input_array_2D, input_seq_length, output_seq_length, batch_size
        )

    def test_input_seq_length_equals_len_input_array(self):
        """Inputs are as they will be in a single-step inference environment:
        output_seq_length = 0, batch_size = 1, and input_seq_length is
        equal to len(array) - the length of the input dataset.

        No need to test if input_seq_length < len(array) since this is default
        behaviour and tested above.
        """
        input_seq_length = 200
        output_seq_length = 0
        batch_size = 1
        # Just contains input_seq_length elements
        array_single_batch = np.arange(input_seq_length).reshape(-1, 1)

        self.compare_feature_and_target_shapes(
            array_single_batch, input_seq_length, output_seq_length, batch_size
        )

    def test_input_is_numpy(self, input_array_2D):
        not_a_numpy_array = list(input_array_2D)

        with pytest.raises(ValueError) as exec_info:
            assert make_tf_dataset(not_a_numpy_array)
