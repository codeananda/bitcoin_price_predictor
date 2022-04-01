#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using close price data, predict the close of Bitcoin on the next day.

@author: king
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

# Size of train and test sets respectively in num. days
TRAIN_NUM_DAYS = 365
TEST_NUM_DAYS = 30
# How many days into the future to predict the close
PREDICT_WINDOW_NUM_DAYS = 1

DOWNLOAD_DIR = Path("../download")

sns.set()

price = pd.read_csv(DOWNLOAD_DIR / "price.csv")
# Just use close price data
close = price.loc[:, "c"]
close_train = close[-TRAIN_NUM_DAYS:-TEST_NUM_DAYS]
close_test = close[-TEST_NUM_DAYS:]

# Define the range of values for training
X_train = close_train[:-PREDICT_WINDOW_NUM_DAYS]
y_train = close_train[PREDICT_WINDOW_NUM_DAYS:]


### NEEDS TO BE PROPERLY DEFINED, DID HE DO IT CORRECTLY IN THE ARTICLE???
#############################
X_test = close_test
#############################

min_max = MinMaxScaler()

# Prep for training
X_train = close_train.values.reshape(-1, 1)
X_train = min_max.fit_transform(X_train)

# We are training to predict the next day's close
# so cut off final val for X_train and start at index 1 for
# y_train
X_train = training_set_std_scale[:-1]
y_train = training_set_std_scale[1:]
X_train = X_train.reshape((-1, 1, 1))

regressor = Sequential([LSTM(4), Dense(1)])
regressor.compile(optimizer="adam", loss="mean_squared_error")
history = regressor.fit(X_train, y_train, batch_size=5, epochs=100)

testing_set = close_test.values.reshape((-1, 1))
testing_set = scaler.transform(testing_set)
testing_set = testing_set.reshape((-1, 1, 1))

predicted_price = regressor.predict(testing_set)
predicted_price = scaler.inverse_transform(predicted_price)

fig, ax = plt.subplots()
plt.plot(close_test.values, c="b", label="BTC Price")
plt.plot(predicted_price, c="r", label="Forecast")
plt.legend()
ax.set(xlabel="Days", ylabel="Price (USD)", title="30 Forcasted BTC Price Last 30 Days")
plt.show()
