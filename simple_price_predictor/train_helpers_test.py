from simple_price_predictor.constants import HISTORICAL_BITCOIN_CSV_FILEPATH
from simple_price_predictor.train_helpers import (
    load_raw_bitcoin_df,
    make_tf_dataset,
    get_optimizer,
    build_LSTM_training,
)
import os
import tensorflow as tf
import numpy as np
from wandb.keras import WandbCallback
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping


import pytest
from pytest_cases import parametrize_with_cases, fixture


DATASET_SIZE = 1000000


@fixture
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


class ZeroBatchCases:
    """Cases for make_tf_dataset to test ValueErrors resulting in creating
    datasets with no elements.

    All return values are of form
    (len_array, input_seq_length, output_seq_length, batch_size)
    """

    def pass_inference_setting_1(self):
        """Ensures functions when creating exactly one batch of data.
        """
        return 100, 100, 0, 1

    def pass_inference_setting_2(self):
        """Ensures functions if want to create many single batches of data
        for inference.
        """
        return 150, 100, 0, 1

    def fail_single_output(self):
        return 100, 100, 1, 1

    def fail_zero_output(self):
        return 100, 100, 0, 2

    def fail_large_batch_size(self):
        return 100, 20, 1, 100

    def fail_input_seq_length_greater_than_len_array(self):
        return 100, 150, 0, 1


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

        with pytest.raises(TypeError) as exec_info:
            assert make_tf_dataset(not_a_numpy_array)

    @parametrize_with_cases(
        "len_array, input_seq_length, output_seq_length, batch_size",
        cases=ZeroBatchCases,
        prefix="fail_",
    )
    def test_num_created_batches_is_0(
        self, len_array, input_seq_length, output_seq_length, batch_size
    ):
        """Tests for the ValueError raised if num_created_batches == 0
        """
        with pytest.raises(ValueError):
            make_tf_dataset(
                np.arange(len_array).reshape(-1, 1),
                input_seq_length,
                output_seq_length,
                batch_size,
            )

    @parametrize_with_cases(
        "len_array, input_seq_length, output_seq_length, batch_size",
        cases=ZeroBatchCases,
        prefix="pass_",
    )
    def test_num_created_batches_is_not_0(
        self, len_array, input_seq_length, output_seq_length, batch_size
    ):
        ds = make_tf_dataset(
            np.arange(len_array).reshape(-1, 1),
            input_seq_length,
            output_seq_length,
            batch_size,
        )
        num_created_batches = len(list(ds.as_numpy_iterator()))
        assert num_created_batches > 0


class OptimizerCases:
    def case_adam(self):
        return "adam"

    def case_rmsprop(self):
        return "rmsprop"

    def case_AdAM(self):
        return "AdAM"

    def case_RMSPRoP(self):
        return "RMSPRoP"


class LRCases:
    def case_default_lr(self):
        return 1e-4

    def case_small_lr(self):
        return 1e-10

    def case_big_lr(self):
        return 100

    def case_very_big_lr(self):
        return 1e10


class LossCases:
    # Keras will not throw an error when compiling the model
    # Only when you try to fit will you get an error if you've passed
    # incompatible loss
    def case_mse(self):
        return "mse"

    def case_mean_squared_error(self):
        return "mean_squared_error"

    def case_mae(self):
        return "mae"


class NumericCases:
    def fail_value_0(self):
        return 0

    def fail_value_negative(self):
        return -5

    def fail_type_decimal(self):
        return 4.5

    def pass_1(self):
        return 1

    def pass_2(self):
        return 2

    def pass_10(self):
        return 10


class Test_get_optimizer:
    @parametrize_with_cases("choice", cases=OptimizerCases)
    @parametrize_with_cases("lr", cases=LRCases)
    def test_runs(self, choice, lr):
        opt = get_optimizer(optimizer=choice, learning_rate=lr)

        opt_dict = {
            "adam": Adam,
            "rmsprop": RMSprop,
        }

        expected_opt = opt_dict[choice.lower()]

        assert isinstance(opt, expected_opt)
        assert opt.learning_rate.numpy() == np.float32(lr)

    def test_only_support_adam_and_rmsprop(self):
        choice = "not adam or rmsprop"

        with pytest.raises(ValueError):
            assert get_optimizer(optimizer=choice)


class Test_build_LSTM_training:
    @parametrize_with_cases("optimizer", cases=OptimizerCases)
    def test_optimizer(self, optimizer):
        model = build_LSTM_training(optimizer=optimizer)

        opt_dict = {
            "adam": Adam,
            "rmsprop": RMSprop,
        }

        expected_opt = opt_dict[optimizer.lower()]
        assert isinstance(model.optimizer, expected_opt)

    @parametrize_with_cases("learning_rate", cases=LRCases)
    def test_learning_rate(self, learning_rate):
        model = build_LSTM_training(learning_rate=learning_rate)

        assert model.optimizer.learning_rate.numpy() == np.float32(learning_rate)

    @parametrize_with_cases("loss", cases=LossCases)
    def test_loss(self, loss):
        model = build_LSTM_training(loss=loss)

        assert model.loss == loss

    @parametrize_with_cases("units", cases=NumericCases, prefix="pass_")
    def test_units_pass(self, units):
        model = build_LSTM_training(units=units)

        # Iterate over all layers apart from output
        for layer in model.layers[:-1]:
            assert layer.units == units

    @parametrize_with_cases("units", cases=NumericCases, prefix="fail_value_")
    def test_units_fail_value(self, units):
        with pytest.raises(ValueError):
            model = build_LSTM_training(units=units)

    @parametrize_with_cases("units", cases=NumericCases, prefix="fail_type_")
    def test_units_fail_type(self, units):
        with pytest.raises(TypeError):
            model = build_LSTM_training(units=units)

    @parametrize_with_cases("batch_size", cases=NumericCases, prefix="pass_")
    def test_batch_size_pass(self, batch_size):
        model = build_LSTM_training(batch_size=batch_size)

        # Iterate over all layers apart from output
        for layer in model.layers[:-1]:
            actual_batch_size = layer.get_config()["batch_input_shape"][0]
            assert actual_batch_size == batch_size

    @parametrize_with_cases("batch_size", cases=NumericCases, prefix="fail_value_")
    def test_batch_size_fail_value(self, batch_size):
        with pytest.raises(ValueError):
            model = build_LSTM_training(batch_size=batch_size)

    @parametrize_with_cases("batch_size", cases=NumericCases, prefix="fail_type_")
    def test_batch_size_fail_type(self, batch_size):
        with pytest.raises(TypeError):
            model = build_LSTM_training(batch_size=batch_size)

    @parametrize_with_cases("timesteps", cases=NumericCases, prefix="pass_")
    def test_timesteps_pass(self, timesteps):
        model = build_LSTM_training(timesteps=timesteps)

        # Iterate over all layers apart from output
        for layer in model.layers[:-1]:
            actual_timesteps = layer.get_config()["batch_input_shape"][1]
            assert actual_timesteps == timesteps

    @parametrize_with_cases("timesteps", cases=NumericCases, prefix="fail_value_")
    def test_timesteps_fail_value(self, timesteps):
        with pytest.raises(ValueError):
            model = build_LSTM_training(timesteps=timesteps)

    @parametrize_with_cases("timesteps", cases=NumericCases, prefix="fail_type_")
    def test_timesteps_fail_type(self, timesteps):
        with pytest.raises(TypeError):
            model = build_LSTM_training(timesteps=timesteps)

    @parametrize_with_cases("num_layers", cases=NumericCases, prefix="pass_")
    def test_num_layers_pass(self, num_layers):
        model = build_LSTM_training(num_layers=num_layers)

        # num_layers is all the LSTM layers you add (not including the output layer)
        assert len(model.layers) == num_layers + 1

    @parametrize_with_cases("num_layers", cases=NumericCases, prefix="fail_value_")
    def test_num_layers_value_error(self, num_layers):
        with pytest.raises(ValueError):
            model = build_LSTM_training(num_layers=num_layers)

    @parametrize_with_cases("num_layers", cases=NumericCases, prefix="fail_type_")
    def test_num_layers_type_error(self, num_layers):
        with pytest.raises(TypeError):
            model = build_LSTM_training(num_layers=num_layers)
