import streamlit as st
import pandas as pd
import numpy as np
import requests

st.title("Predict the Price of Bitcoin in 1 Hour's Time")

if st.button('Get Recent Bitcoin Data'):
    # Get hourly Bitcoin data
    url = "https://api.coincap.io/v2/assets/bitcoin/history?interval=h1"

    payload={}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)

    response.text
    'Success'
