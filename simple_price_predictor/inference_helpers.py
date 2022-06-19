import pandas as pd
import tensorflow as tf
import numpy as np
import os
import pickle
import time
import requests

from simple_price_predictor.secrets import COINCAP_AUTH_HEADER


def get_last_num_days_hourly_bitcoin_data(num_days):
    """Call Coincap API and request last num_days of hourly Bitcoin USD data,
    return DataFrame with 'date' and 'price' columns. Date column is in UTC.

    Returns
    -------
    pd.DataFrame
        Dataframe (columns: 'date', 'price' with correct types).
        Price is rounded to 2 decimal places. Last row contains most recent
        price, first contains price num days ago.
    """
    if not isinstance(num_days, int):
        raise TypeError(
            f"`num_days` must be of type int. Received type {type(num_days)}"
        )
    if not 0 < num_days < 31:
        raise ValueError(
            f"`num_days` must be greater than 0 and less than 31. Received {num_days}"
        )
    num_seconds_in_num_days = 60 * 60 * 24 * num_days
    num_milliseconds_in_num_days = num_seconds_in_num_days * 1000

    now_ns = str(time.time_ns())
    # Take first 13 digits for milliseconds
    # Coincap API only accepts milliseconds
    now_ms = int(now_ns[:13])
    num_days_ago = now_ms - num_milliseconds_in_num_days

    # Get Bitcoin data for last num days
    url = (
        f"https://api.coincap.io/v2/assets/bitcoin/history?interval=h1"
        f"&start={num_days_ago}&end={now_ms}"
    )

    payload = {}
    headers = {"Authorization": COINCAP_AUTH_HEADER}
    response = requests.request("GET", url, headers=headers, data=payload)
    response.raise_for_status()

    json_data = response.json()
    bitcoin_data = json_data["data"]

    df = pd.DataFrame(bitcoin_data)
    df = df.loc[:, ["date", "priceUsd"]]
    df.rename(mapper={"priceUsd": "price"}, inplace=True, axis=1)
    df["date"] = df["date"].apply(pd.to_datetime)
    df["price"] = df["price"].apply(pd.to_numeric)
    df["price"] = df["price"].round(2)
    df.sort_values("date", ascending=False, ignore_index=True, inplace=True)
    return df
