import pytest
from pytest_cases import parametrize_with_cases
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import pickle
import time
import requests

from simple_price_predictor.inference_helpers import (
    get_last_num_days_hourly_bitcoin_data,
)


class NumDaysNumericCases:
    def fail_value_0(self):
        return 0

    def fail_value_neg_5(self):
        return -5

    def fail_value_31(self):
        return 31

    def pass_30(self):
        return 30

    def pass_1(self):
        return 1

    def pass_5(self):
        return 5

    def fail_type_float(self):
        return 3.5

    def fail_type_str(self):
        return "10"


class Test_get_last_num_days_hourly_bitcoin_data:
    @parametrize_with_cases("num_days", cases=NumDaysNumericCases, prefix="fail_value")
    def test_num_days_outside_accepted_range(self, num_days):
        with pytest.raises(ValueError):
            get_last_num_days_hourly_bitcoin_data(num_days)

    @parametrize_with_cases("num_days", cases=NumDaysNumericCases, prefix="pass_")
    def test_num_days_within_accepted_range(self, num_days):
        df = get_last_num_days_hourly_bitcoin_data(num_days)
        #  Give a 2 hour leeway to account for slight API downtime
        assert (num_days * 24 - 2) <= len(df) <= (num_days * 24)

    @parametrize_with_cases("num_days", cases=NumDaysNumericCases, prefix="fail_type")
    def test_num_days_not_int(self, num_days):
        with pytest.raises(TypeError):
            get_last_num_days_hourly_bitcoin_data(num_days)
