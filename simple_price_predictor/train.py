import pandas as pd
import numpy as np
import tensorflow as tf
from fastquant import get_crypto_data

# Data starts 08/2017 but good enough for now.
bitcoin = get_crypto_data("BTC/USDT", "2008-01-01", "2022-05-15")
bitcoin = bitcoin.close

# Processing steps to take
