import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from deploy_helpers import get_last_8_days_hourly_bitcoin_data

sns.set()

st.title("Predict the Price of Bitcoin in 1 Hour's Time")

if st.button("Get Recent Bitcoin Data"):
    bitcoin = get_last_8_days_hourly_bitcoin_data()

    bitcoin

    fig, ax = plt.subplots()
    bitcoin.plot(x="date", y="price", ax=ax, figsize=plt.figaspect(1 / 2))

    # Workaround needed to totally control matplotlib figsize
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)

if st.button("Make Prediction for Next Hour"):
    st.write("Not yet implemented")
