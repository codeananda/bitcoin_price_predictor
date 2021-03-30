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
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.optimizers import Adam

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

DOWNLOAD_DIR = Path('../download')
DATA_DIR = Path('../data')


"""########## LOAD DATA ##########"""

def load_close_data(DOWNLOAD_DIR, dropna=False):
    price = pd.read_csv(DOWNLOAD_DIR / 'price.csv', parse_dates=[0])
    price = price.set_index('timestamp')
    close = price.loc[:, 'c']
    if dropna:
        close = close.dropna()
    data = close.values
    return data


def load_dataset_1():
    with open(DATA_DIR / 'train_1.pkl', 'rb') as f:
        train_1 = pickle.load(f)

    with open(DATA_DIR / 'val_1.pkl', 'rb') as f:
        val_1 = pickle.load(f)

    with open(DATA_DIR / 'test_1.pkl', 'rb') as f:
        test_1 = pickle.load(f)

    return train_1, val_1, test_1


def load_dataset_2():
    with open(DATA_DIR / 'train_2.pkl', 'rb') as f:
        train_2 = pickle.load(f)

    with open(DATA_DIR / 'val_2.pkl', 'rb') as f:
        val_2 = pickle.load(f)

    return train_2, val_2


def get_training_data():
    """
    Convenience function to quickly load in train and val data to train models on.
    """
    # Load in data
    data = load_close_data(DOWNLOAD_DIR, dropna=True)
    # Convert val/test percentages to numbers
    n_val, n_test = get_n_val_and_n_test(data, 0.12, 0.12)
    # Split data
    tvt = train_val_test_split(data, n_val, n_test)
    # Scale data
    train, val, test = scale_train_val_test(*tvt, scaler='log')
    # Get data into form Keras needs
    X_train, X_val, y_train, y_val = transform_to_keras_input(train, val, 168)
    return (X_train, X_val, y_train, y_val)

"""########## SPLIT DATA #############"""

def get_n_test_samples(data, test_size):
    # # data split
    n_test = int(len(data) * test_size)
    return n_test


def get_n_val_and_n_test(data, val_size, test_size):
    n_val = int(len(data) * val_size)
    n_test = int(len(data) * test_size)  
    return n_val, n_test


def train_val_test_split(data, n_val, n_test):
    test = data[-n_test:]
    val = data[-n_test - n_val : -n_test]
    train = data[:-n_test - n_val]
    return train, val, test


def _train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]



"""########## RESHAPE ##########"""

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


# Create train and val sets to input into Keras model
# we do not need test sets at this stage, just care about 
# validation, not testing
def transform_to_keras_input(train, val, n_in):
    # Transform to keras input
    train_data = _series_to_supervised(train, n_in=n_in)
    val_data = _series_to_supervised(val, n_in=n_in)
    # Create X and y variables
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_val, y_val = val_data[:, :-1], val_data[:, -1]
    return X_train, X_val, y_train, y_val

"""########## SCALE ##########"""

# I can easily add more functionality to this by writing
# other functions like min_max_scale_train_val_test()
# and add them under each 'if scaler == 'min_max': 
def scale_train_val_test(train, val, test=None, scaler='log'):
    if scaler.lower() == 'log':
        train, val, test = _scale_log(train, val, test)
    elif scaler.lower() == 'log_and_divide_20':
        train, val, test = _scale_log_and_divide(train, val, test, 20)
    elif scaler.lower() == 'log_and_divide_15':
        train, val, test = _scale_log_and_divide(train, val, test, 15)
    elif scaler.lower().startswith('log_and_range'):
        train, val, test = _scale_log_and_range(train, val, test, scaler)
    else:
        raise Exception('''Please enter a supported scaling type: log, log_and_divide_20
                        log_and_divide_15, log_and_range_a_b (where [a, b] is the range
                        you want to scale to.''')
    return train, val, test


# Scaling just on train and val sets (since test is unnecessary for training)
def scale_train_val(train, val, scaler='log'):
    if scaler.lower() == 'log':
        train, val = _scale_log(train, val)
    elif scaler.lower() == 'log_and_divide_20':
        train, val = _scale_log_and_divide(train, val, divisor=20)
    elif scaler.lower() == 'log_and_divide_15':
        train, val = _scale_log_and_divide(train, val, divisor=15)
    elif scaler.lower().startswith('log_and_range'):
        train, val = _scale_log_and_range(train, val, scaler)
    else:
        raise Exception('''Please enter a supported scaling type: log, log_and_divide_20
                        log_and_divide_15, log_and_range_a_b (where [a, b] is the range
                        you want to scale to.''')
    return train, val


def _scale_log(train, val, test=None):
    train = np.log(train)
    val = np.log(val)
    if test is not None:
        test = np.log(test)
        return train, val, test
    return train, val


def _scale_log_and_divide(train, val, divisor=20):
    # Take log
    train, val = _scale_log(train, val)
    # Divide by divisor
    train /= divisor
    val /= divisor
    return train, val



# Taken from this SO answer: https://tinyurl.com/j5rppewr
def _scale_to_range(seq, a, b, min=None, max=None):
    """
    Given a sequence of numbers - seq - scale all of its values to the 
    range [a, b]. 

    Default behaviour will map min(seq) to a and max(seq) to b.
    To override this, set max and min yourself.
    """
    assert a < b
    # Default is to use the max of the seq as the min/max
    # Can override this and input custom min and max values
    # if, for example, want to scale to ranges not necesarily included
    # in the data (as in our case with the train and val data)
    if max is None:
        max = max(seq)
    if min is None:
        min = min(seq)
    assert min < max
    def scaled_val(val, a, b, minimum, maximum):
        # Scale val into [a, b] given the max and min values of the seq
        # it belongs to.
        # Taken from this SO answer: https://tinyurl.com/j5rppewr
        numerator = (b - a) * (val - minimum)
        denominator = maximum - minimum
        return (numerator / denominator) + a

    scaled_seq = np.array([scaled_val(val, a, b, min, max) \
                           for val in seq])

    return scaled_seq


def _scale_log_and_range(train, val, scaler):
    train, val, _ = _scale_log(train, val)
    # Split scaler on underscores to extract the min and max values for the range
    elements = scaler.split('_')
    # Calc args for _scale_to_range
    range_bottom = float(elements[-2])
    range_top = float(elements[-1])
    min_value = min(train)
    max_value = max(val)
    args = [range_bottom, range_top, min_value, max_value]
    train = _scale_to_range(train, *args)
    val = _scale_to_range(val, *args)  
    return train, val  


def inverse_scale(data, scaler='log'):
    if scaler.lower() != 'log':
        raise TypeError("Only 'log' scaling supported at this time." )
    # Inverse log scale
    inverse_scaled_data = [np.exp(d) for d in data]
    return inverse_scaled_data


"""########## PLOT ##########"""

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


def plot_train_val_test(train, val, test):
    fig, ax = plt.subplots()
    ax.plot(train, 'b', label='Train')
    ax.plot([None for x in train] + [x for x in val], 'r', label='Val')
    ax.plot([None for x in train] + [None for x in val] + [x for x in test],
            'g', label='Test')
    ax.legend()
    plt.show()


"""########## MEASURE ##########"""
# root mean squared error, or rmse
def _measure_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

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


"""########## MODEL BUILD AND FIT ##########"""

def build_model(config):
    model = Sequential([
        Dense(config.n_nodes, activation=config.activation,
              input_dim=config.n_input),
        Dense(config.n_nodes, activation=config.activation),
        Dense(config.n_nodes, activation=config.activation),
        Dense(1)
    ])
    learning_rate_schedule = InverseTimeDecay(config.initial_lr,
                                              config.decay_steps,
                                              config.decay_rate)
    optimizer = Adam(learning_rate_schedule)
    model.compile(loss=config.loss, 
                  optimizer=optimizer,
                  metrics=[RootMeanSquaredError()])
    return model


def fit_model(model, config, X_train, X_val, y_train, y_val):
    history = model.fit(
                X_train, 
                y_train, 
                epochs=config.n_epochs,
                batch_size=config.n_batch, 
                verbose=config.verbose,
                shuffle=False, 
                validation_data=(X_val, y_val),
                callbacks=[WandbCallback()])
    return history


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


"""########### EVALUATE ##########"""
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


def get_preds_and_rmse(model, X_train, X_val, y_train, y_val):
    # Calculate predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # Calculate rmse for train and val data
    eval_results_train = model.evaluate(X_train, y_train, verbose=0)
    eval_results_val = model.evaluate(X_val, y_val, verbose=0)
    rmse_train = eval_results_train[1]
    rmse_val = eval_results_val[1]

    return y_pred_train, y_pred_val, rmse_train, rmse_val

"""########## WANDB ##########"""
def upload_history_to_wandb(history):
    # Turn into df
    history_df = pd.DataFrame.from_dict(history.history)
    # Turn into wandb Table
    history_table = wandb.Table(dataframe=history_df)
    # Log
    wandb.log({'history': history_table})


"""########## FULL PROCESS ##########"""
def train_and_validate(config, dataset=1):
    # Load data
    if dataset == 1:
        train, val, _ = load_dataset_1()
    elif dataset == 2:
        train, val = load_dataset_2()
    else:
        raise Exception('Only two datasets are available: 1 or 2')
    # Scale data
    train_scaled, val_scaled = scale_train_val(train, val, scaler=config.scaler)
    # Get data into form Keras needs
    X_train, X_val, y_train, y_val = transform_to_keras_input(train_scaled,
                                                              val_scaled,
                                                              config.n_input)
    # Build and fit model
    model = build_model(config)
    history = fit_model(model, config, X_train, X_val, y_train, y_val)
    # Plot loss, rmse, and 1-rmse curves
    plot_metric(history, metric='loss', start_epoch=10)
    plot_metric(history, metric='root_mean_squared_error', start_epoch=10)
    plot_metric(history, metric='1-root_mean_squared_error', start_epoch=10)
    # Store history on wandb
    upload_history_to_wandb(history)

    # Calc preds and pred rmse on train and val datasets
    preds_and_rmse = get_preds_and_rmse(model, 
                                        X_train, 
                                        X_val, 
                                        y_train, 
                                        y_val)
    # y_pred_train, y_pred_val, rmse_train, rmse_val
     
    # Turn all predictions and rmse into a log scale (for easy comparison
    # between different scalings)
    if config.scaler.lower() != 'log':
        y_pred_train_log, y_pred_val_log, \
            rmse_train_log, rmse_val_log = convert_to_log(*preds_and_rmse)

    # EVERYTHING IN HERE NEEDS TO BE CONVERTED TO LOG.
    # Plot predictions for train and val data
    _plot_actual_vs_pred(y_train, y_pred_train, rmse=rmse_train,
                         name='X_train preds', logy=True)
    _plot_actual_vs_pred(y_val, y_pred_val, rmse=rmse_val,
                         name='X_val preds', logy=True)
    _plot_preds_grid(y_train, y_pred_train, rmse_train)
    return history


# Log - do nothing
# 
    def convert_to_log(values, scaler):
        pac