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
    get_raw_coincap_bitcoin_data,
    process_coincap_response,
)

MOCK_TIMESTAMP = 42


class MockCoincapAPIResponse:
    @staticmethod
    def raise_for_status():
        return True

    @staticmethod
    def json():
        data = []
        # Choose 1 full day of data for simplicity
        for i in range(1, 25):
            row = {
                "priceUsd": f"{i * 10.}",  #  Must be float
                "time": i,  # Dropped in preprocessing
                "circulatingSupply": f"{i}",  # Dropped during processing
                "date": f"2022-06-18T{i-1:0>2}:00:00.000Z",  # Must be datetime
            }
            data.append(row)
        response = {
            "data": data,
            "timestamp": MOCK_TIMESTAMP,  # Dropped in preprocessing
        }
        return response


# monkeypatched requests.get moved to a fixture
@pytest.fixture
def mock_coincap_api_response(monkeypatch):
    """requests.get() mocked to return MockCoincapAPIResponse().json()
    """

    def mock_get(*args, **kwargs):
        return MockCoincapAPIResponse()

    monkeypatch.setattr(requests, "get", mock_get)


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
            get_raw_coincap_bitcoin_data(num_days)

    @parametrize_with_cases("num_days", cases=NumDaysNumericCases, prefix="pass_")
    def test_num_days_within_accepted_range(self, num_days, mock_coincap_api_response):
        result = get_raw_coincap_bitcoin_data(num_days)
        assert result["timestamp"] == MOCK_TIMESTAMP

    @parametrize_with_cases("num_days", cases=NumDaysNumericCases, prefix="fail_type")
    def test_num_days_not_int(self, num_days):
        with pytest.raises(TypeError):
            get_raw_coincap_bitcoin_data(num_days)
