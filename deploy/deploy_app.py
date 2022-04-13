import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from io import BytesIO
import plotly.express as px
import tensorflow as tf

from deploy_helpers import get_last_8_days_hourly_bitcoin_data

st.title("Predict the Price of Bitcoin in 1 Hour's Time")

if st.button("Get Recent Bitcoin Data"):
    bitcoin = get_last_8_days_hourly_bitcoin_data()

    bitcoin

    fig = px.line(bitcoin, x="date", y="price")

    st.plotly_chart(fig)

if st.button("Make Prediction for Next Hour"):
    pass
