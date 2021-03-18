import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from tqdm.notebook import trange, tqdm

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import RootMeanSquaredError

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

import wandb
from wandb.keras import WandbCallback

# Run this part in notebook to configure experiment
"""
# Initalize wandb with project name
run = wandb.init(project='bitcoin_price_predictor',
                 config={
                     'test_size': 0.15,
                     'n_input': 168, # num lag observations
                     'n_nodes': 300, # num nodes per lauyer
                     'n_epochs': 100, # num training epochs
                     'n_batch': 168 * 20, # batch size
                     'n_repeats': 5, # num repeats of WF validation
                     'activation': 'relu',
                     'loss': 'mse',
                     'optimizer': 'adam',
                     'val_split': 0.15,
                     'verbose': 0, # control verbosity of Keras fit
                     'dropna': True # whether to drop missing values from data
                         })

config = wandb.config # use this to configure experiment
"""


def _train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


def _series_to_supervised(data, n_in=1, n_out=1):
    df = pd.DataFrame(data)
    cols = []
    # input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    agg.dropna(inplace=True)
    return agg.values

# root mean squared error, or rmse
def _measure_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


# Define, compile and fit a model on train and val data
def _model_fit(train, config):
    # prepare data
    train_data = _series_to_supervised(train, n_in=config.n_input)
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    # define model
    model = Sequential()
    model.add(Dense(config.n_nodes, 
                    activation=config.activation, 
                    input_dim=config.n_input))
    model.add(Dense(1))
    # compile
    model.compile(loss=config.loss, optimizer=config.optimizer)
    # fit
    history = model.fit(
                X_train, 
                y_train, 
                epochs=config.n_epochs,
                batch_size=config.n_batch, 
                verbose=config.verbose,
                shuffle=False, 
                validation_split=config.val_split,
                callbacks=[WandbCallback()]
                )
    return (model, history)

# forecast with a pre-fit model
def _model_predict(model, history, config):
    # unpack config
    # n_input, _, _, _ = config
    # prepare data
    x_input = np.array(history[-config.n_input:]).reshape(1, config.n_input)
    # forecast, one at a time
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]

# walk-forward validation for univariate data
def _walk_forward_validation(data, n_test, config):
    predictions = []
    # split dataset
    train, test = _train_test_split(data, n_test)
    # fit model
    model = _model_fit(train, config)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test dataset
    for i in trange(len(test)):
        # use fitted model to make forecast
        yhat = _model_predict(model, history, config)
        # store forecaxst in the list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = _measure_rmse(test, predictions)
    print(' > %.3f' % error)
    return error, predictions, test


def _plot_actual_vs_pred(y_true, y_pred, rmse, repeat):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(y_true, 'b', label='Test data')
    ax.plot(y_pred, 'r', label='Preds')
    ax.legend()
    ax.set(xlabel='Hours', ylabel='BTC Price ($)',
           title=f'Actuals vs. Preds - RMSE {rmse:.3f} - Repeat #{repeat}')
    wandb.log({f'Actuals vs. Preds #{repeat}': wandb.Image(fig)})
    plt.show()


def _plot_actual_vs_all_preds(y_true, y_preds, rmse_scores):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(y_true, 'b', label='Test data')
    for i, y_pred in enumerate(y_preds):
        ax.plot(y_pred, label=f'Preds #{i} - RMSE {rmse_scores[i]:.3f}')
    ax.legend()
    ax.set(xlabel='Hours', ylabel='BTC Price ($)',
            title='Actuals vs. All Preds')
    wandb.log({f'Actuals vs. All Preds': wandb.Image(fig)})
    plt.show()


def load_close_data(DOWNLOAD_DIR, dropna=False):
    price = pd.read_csv(DOWNLOAD_DIR / 'price.csv', parse_dates=[0])
    price = price.set_index('timestamp')
    close = price.loc[:, 'c']
    if dropna:
        close = close.dropna()
    data = close.values
    return data


def get_n_test_samples(data, test_size):
    # # data split
    n_test = int(len(data) * test_size)
    return n_test

# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=30, plot=True):
    scores = []
    predictions = []
    for i in range(n_repeats):
        print(f'Repeat #{i}')
        score, pred, test = _walk_forward_validation(data, n_test, config)
        scores.append(score)
        predictions.append(pred)
        if plot:
            _plot_actual_vs_pred(test, pred, score, i)
    if plot:
        _plot_actual_vs_all_preds(test, predictions, scores)
    wandb.log({'Repeated walk-forward validation scores': scores})
    return scores

# Summarize model performance
def summarize_scores(name, scores):
    # print a summary
    scores_m, scores_std = np.mean(scores), np.std(scores)
    ax_title = f'{name}: {scores_m:.3f} RMSE (+/- {scores_std:.3f})'
    print(ax_title)
    log_title = 'RMSE Walk-Forward Validation Scores Distribution'
    fig, ax = plt.subplots()
    sns.boxplot(x=scores, ax=ax)
    ax.set(xlabel='RMSE', title=ax_title)
    wandb.log({log_title: wandb.Image(fig)})
    plt.show()

