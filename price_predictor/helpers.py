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
    model.compile(loss=config.loss, 
                  optimizer=config.optimizer,
                  metrics=[RootMeanSquaredError()])
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


def _plot_actual_vs_pred(y_true, y_pred, rmse=None, repeat=None, name=None,
                         logy=False):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(y_true, 'b', label='Test data')
    ax.plot(y_pred, 'r', label='Preds')
    ax.legend()

    if rmse is not None and repeat is not None:
        fig_title = f'Actuals vs. Preds - RMSE {rmse:.5f} - Repeat #{repeat}'
        log_title = f'Actuals vs. Preds #{repeat}'
    elif rmse is not None and repeat is None:
        fig_title = f'Actuals vs. Preds - {name} - RMSE {rmse:.5f}'
        log_title = f'Actuals vs. Preds - {name}'
    elif rmse is None and repeat is not None:
        raise Exception('Cannot enter repeat on its own')
    else:
        fig_title = f'Actuals vs. Preds - {name}'
        log_title = fig_title

    ylabel = 'BTC Price ($)'
    if logy:
        ylabel = 'log(BTC Price USD)'

    ax.set(xlabel='Hours', ylabel=ylabel,
           title=fig_title)
    wandb.log({log_title: wandb.Image(fig)})
    plt.show()


def _plot_preds_grid(y_true, y_pred, rmse):
    """
    Built to make a 2x4 grid of preds vs actuals for X_train
    """
    fig = plt.figure(figsize=(20, 10))
    # Plot full predictions
    if len(y_true) > 80000:
        plt.subplot(2, 5, 1)
    else:
        plt.subplot(2, 4, 1)
    plt.plot(y_true, 'b', label='Actual')
    plt.plot(y_pred, 'r', label='Preds')
    plt.legend()
    if len(y_true) > 80000:
        plt.xticks(np.arange(0, 100000, 20000))
    else:
        plt.xticks(np.arange(0, 84000, 14000))
    # Plot predictions for each 10k hours
    for i in range(0, len(y_true) // 10000 + 1):
        if len(y_true) > 80000:
            plt.subplot(2, 5, 2+i)
        else:
            plt.subplot(2, 4, 2+i)
        plt.plot(y_true[i * 10000: (i+1) * 10000], 'b')
        plt.plot(y_pred[i * 10000: (i+1) * 10000], 'r')
        plt.xticks(ticks=np.arange(0, 12000, 2000),
                   labels=np.arange(i * 10000, (i+1) * 10000 + 2000, 2000))
        if i == 1:
            title = f'X_train predictions (broken down) - X_train RMSE {rmse:.5f}'
            plt.title(title)
    plt.tight_layout()
    log_title = 'X_train predictions (broken down)'
    wandb.log({log_title: wandb.Image(fig)})
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


def plot_metric(history, metric='loss', ylim=None, start_epoch=0):
    """
    * Given a Keras history, plot the specific metric given. 
    * Can also plot '1-metric'
    * Set the y-axis limits with ylim
    * Since you cannot know what the optimal y-axis limits will be ahead of time,
      set the epoch you will start plotting from (start_epoch) to avoid plotting
      the massive spike that these curves usually have at the start and thus rendering
      the plot useless to read.
    """
    # Define here because we need to remove '1-' to calculate the right
    title = f'{metric.title()} - Training and Validation'
    ylabel = f'{metric.title()}'
    is_one_minus_metric = False

    if metric.startswith('1-'):
        # i.e. we calculate and plot 1 - metric rather than just metric
        is_one_minus_metric = True
        metric = metric[2:]
    metric = metric.lower()

    fig, ax = plt.subplots()
    num_epochs_trained = len(history.history[metric])
    epochs = range(1, num_epochs_trained + 1)

    values = history.history[metric]
    val_values = history.history[f'val_{metric}']

    if is_one_minus_metric:
        values = 1 - np.array(history.history[metric])
        val_values = 1 - np.array(history.history[f'val_{metric}'])
    else:
        values = history.history[metric]
        val_values = history.history[f'val_{metric}']

    ax.plot(epochs[start_epoch:], values[start_epoch:], 'b', label='Training')
    ax.plot(epochs[start_epoch:], val_values[start_epoch:], 'r', label='Validation')
        
    ax.set(title=title,
           xlabel='Epoch',
           ylabel=ylabel,
           ylim=ylim)
    ax.legend()
    wandb.log({title: wandb.Image(fig)})
    plt.show()
