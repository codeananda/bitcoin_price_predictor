import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


def get_last_8_days_hourly_bitcoin_data():
    """Call Coincap API and request last 8 days of hourly Bitcoin USD data,
    return DataFrame with 'date' and 'price' columns. Date column is in UTC.

    Returns
    -------
    pd.DataFrame
        Dataframe (columns: 'date', 'price' with correct types).
        Price is rounded to 2 decimal places. Last row contains most recent
        price, first contains price 8 days ago.
    """
    num_seconds_in_8_days = 60 * 60 * 24 * 8
    num_milliseconds_in_8_days = num_seconds_in_8_days * 1000

    now_ns = str(time.time_ns())
    # Take first 13 digits for milliseconds
    # Coincap API only accepts milliseconds
    now_ms = int(now_ns[:13])
    eight_days_ago = now_ms - num_milliseconds_in_8_days

    # Get Bitcoin data for last 8 days
    url = (
        f"https://api.coincap.io/v2/assets/bitcoin/history?interval=h1"
        f"&start={eight_days_ago}&end={now_ms}"
    )

    payload = {}
    headers = {"Authorization": "Bearer bff099f6-aec1-4e2f-8cec-57f8eea14e27"}
    response = requests.request("GET", url, headers=headers, data=payload)
    response.raise_for_status()
    response.status_code

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
