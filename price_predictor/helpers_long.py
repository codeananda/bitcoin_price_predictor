"""
This file contains all functions I wrote over the course of completing this
project. Not all of them were used in the end.

See helpers.py for the functions that I ended up using.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import seaborn as sns
from tqdm.notebook import trange, tqdm

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers.schedules import InverseTimeDecay, ExponentialDecay
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

import wandb
from wandb.keras import WandbCallback

# Run this part in notebook to configure experiment
"""
# Initalize wandb with project name
run = wandb.init(project='bitcoin_price_predictor',
                config={
                    ### Notebook environment
                    'notebook': 'colab', # colab or local
                    ### Data preparation
                    'dataset': 1,
                    'scaler': 'log_and_range_0_1', # log, log_and_divide_a, log_and_range_a_b
                    'n_input': 168, # num lag observations
                    ### Model build
                    'model_type': 'MLP',
                    'activation': 'relu',
                    'loss': 'mse',
                    'optimizer': 'adam', # adam, rmsprop
                    #### LR scheduler and optimizer
                    'use_lr_scheduler': True,
                    'lr_scheduler': 'custom', # InverseTimeDecay, ExponentialDecay, cusom
                    'initial_lr': 1e-4,
#                    'lr': 1e-4, # if use_lr_scheduler == False for fixed LR
                    ### Model fit
                    'n_epochs': 150, # num training epochs
                    'n_batch': 168 * 9, # batch size
                    'verbose': 1, # verbosity of  fit
                    ### EarlyStopping callback
                    'patience': 10,
                    'restore_best_weights': True,
                    'early_stopping_baseline': None, # set to None if there isn't one
                    # Plots
                    'start_plotting_epoch': 0
                        })
config = wandb.config # we use this to configure our experiment
"""


def get_download_and_data_dirs(config):
    """
    Return DOWNLOAD_DIR and DATA_DIR depending on the type of notebook being
    used.

    Google Colab notebooks require different paths to local ones.

    Parameters
    ----------
    config : WandB Config
        Config file to control wandb experiments. Set config.notebook to
        either 'local' or 'colab'.

    Returns
    -------
    DOWNLOAD_DIR, DATA_DIR : tuple of Path objects
        Filepaths to the download and data directories respectively
    """
    if config.notebook.lower() == "colab":
        DOWNLOAD_DIR = Path(
            "/content/drive/MyDrive/1 Projects/bitcoin_price_predictor/download"
        )
        DATA_DIR = Path(
            "/content/drive/MyDrive/1 Projects/bitcoin_price_predictor/data"
        )
    elif config.notebook.lower() == "local":
        DOWNLOAD_DIR = Path("../download")
        DATA_DIR = Path("../data")
    else:
        raise ValueError(
            """Set config.notebook to a supported notebook type:
                        colab or local"""
        )
    return DOWNLOAD_DIR, DATA_DIR


"""########## LOAD DATA ##########"""


def load_close_data(DOWNLOAD_DIR, dropna=False):
    price = pd.read_csv(DOWNLOAD_DIR / "price.csv", parse_dates=[0])
    price = price.set_index("timestamp")
    close = price.loc[:, "c"]
    if dropna:
        close = close.dropna()
    data = close.values
    return data


def load_dataset_1(config):
    """Load dataset 1 as defined in data/define_datasets_1_and_2.png

    Parameters
    ----------
    config : WandB Config
        Config file to control wandb experiments

    Returns
    -------
    train_1, val_1, test_1 : Numpy arrays
        Numpy arrays containing the train/val/test univariate Bitcoin close
        price datasets as defined in data/define_datasets_1_and_2.png
    """
    _, DATA_DIR = get_download_and_data_dirs(config)
    with open(DATA_DIR / "train_1.pkl", "rb") as f:
        train_1 = pickle.load(f)
        if config.drop_first_900_train_elements:
            train_1 = train_1[900:]

    with open(DATA_DIR / "val_1.pkl", "rb") as f:
        val_1 = pickle.load(f)

    with open(DATA_DIR / "test_1.pkl", "rb") as f:
        test_1 = pickle.load(f)

    return train_1, val_1, test_1


def load_dataset_2(config):
    """Load dataset 2 as defined in data/define_datasets_1_and_2.png

    Parameters
    ----------
    config : WandB Config
        Config file to control wandb experiments

    Returns
    -------
    train_2, val_2 : Numpy arrays
        Numpy arrays containing the train/val univariate Bitcoin close
        price datasets as defined in data/define_datasets_1_and_2.png

        Note: there is no test dataset included due to the limited amount of
        data available after train_2 ends.
    """
    _, DATA_DIR = get_download_and_data_dirs(config)
    with open(DATA_DIR / "train_2.pkl", "rb") as f:
        train_2 = pickle.load(f)

    with open(DATA_DIR / "val_2.pkl", "rb") as f:
        val_2 = pickle.load(f)

    return train_2, val_2


def load_train_and_val_data(config):
    """Convenience function to load just the train and val datasets from
    either dataset 1 or 2 (as controlled by the config)

    Parameters
    ----------
    config : WandB Config
        Config file to control wandb experiments

    Returns
    -------
    train, val : Numpy arrays
        Numpy arrays containing the train/val univariate Bitcoin close
        datasets as defined in data/define_datasets_1_and_2.png
    """
    if config.dataset == 1:
        train, val, _ = load_dataset_1(config)
    elif config.dataset == 2:
        train, val = load_dataset_2(config)
    else:
        raise ValueError("Set config.dataset to a supported dataset: 1 or 2")
    return train, val


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
    train = data[: -n_test - n_val]
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


def remove_excess_elements(config, array, is_X=False):
    """
    Take a NumPy array and return it but with the elements removed that
    were not included in training.

    This is for training with RNNs which require all batches to be the exact
    same length. During training, we convert numpy arrays to tf.data.Dataset
    objects, put them in batches and drop the excess elements.

    In this function we do the same process but then transform the tf.data.Dataset
    back into a NumPy array for use later on.

    Note: I feel like there must be a better way to do this. Or I am doing
          this unnecessarily and am using the tf.data.Dataset API incorrectly.

    Note 2: I included is_X as a kwarg because I thought I needed to remove
            values from X in train_and_validate but I actually don't need to.

    """
    # Transform to tf.data.Dataset
    a_ds = tf.data.Dataset.from_tensor_slices(array)
    # Put into batches and drop excess elements
    a_batch = a_ds.batch(config.n_batch, drop_remainder=True)
    # Turn back into a list
    a_list = list(a_batch.as_numpy_iterator())
    # Turn into 2D numpy array (this is 2D becuase the data is batches and has
    # len equal to the number of batches passed to the model during training
    # also equivalent to the number of epochs per training round.
    a_numpy = np.array(a_list)
    # Turn into 1D numpy array
    a_flat = a_numpy.ravel()
    if is_X:
        # Reshape to (samples, timesteps, features) if it's an X array
        a_X_shaped = a_flat.reshape(-1, config.n_input, 1)
        return a_X_shaped
    return a_flat


# Create train and val sets to input into Keras model
# we do not need test sets at this stage, just care about
# validation, not testing
def transform_to_keras_input(config, train, val, n_in):
    """
    Given train and val datasets of univariate timeseries, transform them into sequences
    of length n_in and split into X_train, X_val, y_train, y_val.

    If model is an LSTM, remove the excess elements that occur when arranging data
    into batches (each batch fed into an RNN must be exactly the same length).

    I've chosen to remove the batches here and keep everything as NumPy arrays for
    simplicity. It may be better to work with tf.data.Datasets in general
    e.g.
    >>> train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    >>> train_dataset = train_dataset.repeat().batch(config.n_batch, drop_remainder=True)
    >>> model.fit(train_dataset, ...)
    But this is my first project of this size and I am sticking to what I know.
    Would be interesting to see what performance gains there would be for using
    tf.data.Dataset all the time.

    Ouputs: numpy arrays
    """
    # Transform to keras input
    train_data = _series_to_supervised(train, n_in=n_in)
    val_data = _series_to_supervised(val, n_in=n_in)
    # Create X and y variables
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_val, y_val = val_data[:, :-1], val_data[:, -1]
    if config.model_type.upper() == "LSTM":
        # Remove excess elements in the final batch.
        X_train = remove_excess_elements(config, X_train, is_X=True)
        X_val = remove_excess_elements(config, X_val, is_X=True)
        y_train = remove_excess_elements(config, y_train)
        y_val = remove_excess_elements(config, y_val)
    return X_train, X_val, y_train, y_val


"""########## SCALE ##########"""

# I can easily add more functionality to this by writing
# other functions like min_max_scale_train_val_test()
# and add them under each 'if scaler == 'min_max':
def scale_train_val_test(train, val, test=None, scaler="log"):
    """
    WARNING: MAY NOT WORK
    Written when I thought I would want to scale train, val and test datasets at the same time.
    Turned out, I just wanted to scale train and val all the time, so I wrote scale_train_val
    instead.
    """
    if scaler.lower() == "log":
        train, val, test = _scale_log(train, val, test)
    elif scaler.lower() == "log_and_divide_20":
        train, val, test = _scale_log_and_divide(train, val, test, 20)
    elif scaler.lower() == "log_and_divide_15":
        train, val, test = _scale_log_and_divide(train, val, test, 15)
    elif scaler.lower().startswith("log_and_range"):
        train, val, test = _scale_log_and_range(train, val, test, scaler)
    else:
        raise Exception(
            """Please enter a supported scaling type: log, log_and_divide_20
                        log_and_divide_15, log_and_range_a_b (where [a, b] is the range
                        you want to scale to."""
        )
    return train, val, test


# Scaling just on train and val sets (since test is unnecessary for training)
def scale_train_val(train, val, scaler="log"):
    """Scaled the train and validation datasets based on the scaler type
    specified.

    Parameters
    ----------
    train : np.ndarray
        Training dataset
    val : np.ndarray
        Validation dataset
    scaler: str, optional {'log', 'log_and_divide_a', 'log_and_range_a_b'}
        Scaling to apply to train and validation datasets. Options:

            * 'log' - apply a log transforma
            * 'log_and_divide_a' - first apply a log transform, then divide by
               a (a can be any numeric value)
            * 'log_and_range_a_b' - first apply a log transform, then
               scale the dataset to be in the range [a, b]
               (a and b can be any numeric value but you must have a < b)

        Note: a and b can be any numeric value e.g. 'log_and_divide_20'
        applies a log transformation, then divides the datasets by 20.

    Returns
    -------
    train, val : Tuple of numpy arrays
        Scaled copies of input train and val as dictated by scaler.
    """
    if scaler.lower() == "log":
        train, val = _scale_log(train, val)
    elif scaler.lower().startswith("log_and_divide"):
        train, val = _scale_log_and_divide(train, val, scaler)
    elif scaler.lower().startswith("log_and_range"):
        train, val = _scale_log_and_range(train, val, scaler)
    else:
        raise ValueError(
            """Please enter a supported scaling type: log, log_and_divide_a
                        (first take log, then divide by a), or log_and_range_a_b (first take
                        log then scale to range [a, b])."""
        )
    return train, val


def _scale_log(train, val, test=None):
    """Apply a log transform to train, val and test and return them.

    Parameters
    ----------
    train : np.ndarray
        Training dataset
    val : np.ndarray
        Validation dataset
    test : np.ndarray, optional
        Test dataset, by default None

    Returns
    -------
    train, val, [test] : tuple of np.ndarrays
        Log scaled versions of train, val and test.
    """
    train_log = np.log(train)
    val_log = np.log(val)
    if test is not None:
        test_log = np.log(test)
        return train_log, val_log, test_log
    else:
        return train_log, val_log


def _scale_log_and_divide(train, val, scaler):
    # Take log
    train, val = _scale_log(train, val)
    # Get divisor (last elt of str)
    divisor = scaler.split("_")[-1]
    # Divide by divisor
    train /= divisor
    val /= divisor
    return train, val


def _scale_one_value(value, scaled_min, scaled_max, global_min, global_max):
    """Scale value to the range [scaled_min, scaled_max]. The min/max
    of the sequence/population that value comes from are global_min and
    global_max.

    Parameters
    ----------
    value : int or float
        Single number to be scaled
    scaled_min : int or float
        The minimum value that value can be mapped to.
    scaled_max : int or float
        The maximum value that value can be mapped to.
    global_min : int or float
        The minimum value of the population value comes from. Value must
        not be smaller than this.
    global_max : int or float
        The maximum value of the population value comes from. Value must
        not be bigger than this.

    Returns
    -------
    scaled_value: float
        Value mapped to the range [scaled_min, scaled_max] given global_min
        and global_max values.
    """
    assert value >= global_min
    assert value <= global_max
    # Math adapted from this SO answer: https://tinyurl.com/j5rppewr
    numerator = (scaled_max - scaled_min) * (value - global_min)
    denominator = global_max - global_min
    scaled_value = (numerator / denominator) + scaled_min
    return scaled_value


def _scale_seq_to_range(seq, scaled_min, scaled_max, global_min=None, global_max=None):
    """Given a sequence of numbers - seq - scale its values to the range
    [scaled_min, scaled_max].

    Default behaviour maps min(seq) to scaled_min and max(seq) to
    scaled_max. To map different values to scaled_min/max, set global_min
    and global_max yourself. Manually controlling the global_min/max
    is useful if you map multiple sequences to the same range but each
    sequence does not contain the same min/max values.

    Parameters
    ----------
    seq : 1D array
        1D array containing numbers.
    scaled_min : int or float
        The minimum value of seq after it has been scaled.
    scaled_max : int or float
        The maximum value of seq after it has been scaled.
    global_min : int or float, optional
        The minimum possible value for elements of seq, by default None.
        If None, this is taken to be min(seq). You will want to set
        global_min manually if you are scaling multiple sequences to the
        same range and they don't all contain global_min.
    global_max : int or float, optional
        The maximum possible value for elements of seq, by default None.
        If None, this is taken to be max(seq). You will want to set
        global_max manually if you are scaling multiple sequences to the
        same range and they don't all contain global_max.

    Returns
    -------
    scaled_seq: 1D np.ndarray
        1D array with all values mapped to the range [scaled_min, scaled_max].
    """
    assert seq.ndim == 1
    assert scaled_min < scaled_max
    assert global_min < global_max

    if global_max is None:
        global_max = np.max(seq)
    if global_min is None:
        global_min = np.min(seq)

    scaled_seq = np.array(
        [
            _scale_one_value(value, scaled_min, scaled_max, global_min, global_max)
            for value in seq
        ]
    )

    return scaled_seq


def _scale_log_and_range(train, val, scaler):
    train_log, val_log = _scale_log(train, val)
    # Split scaler on underscores to extract the min and max values for the range
    elements = scaler.split("_")
    # Calculate scaling parameters for _scale_seq_to_range
    scaled_min = float(elements[-2])
    scaled_max = float(elements[-1])
    if not scaled_min < scaled_max:
        raise ValueError(
            f"""You are trying to scale to the range [a, b] where
                        a = {scaled_min} and b = {scaled_max}. Please choose
                        different values such that a < b."""
        )

    global_min_value = min(train_log)
    global_max_value = max(val_log)
    args = [scaled_min, scaled_max, global_min_value, global_max_value]
    train_scaled = _scale_seq_to_range(train_log, *args)
    val_scaled = _scale_seq_to_range(val_log, *args)
    return train_scaled, val_scaled


# Delete if unused in train_and_validate()
def inverse_scale(data, scaler="log"):
    if scaler.lower() != "log":
        raise TypeError("Only 'log' scaling supported at this time.")
    # Inverse log scale
    inverse_scaled_data = [np.exp(d) for d in data]
    return inverse_scaled_data


def convert_to_log(values, scaler, train, val):
    """
    values = [y_pred_train, y_pred_val, rmse_train, rmse_val]
    y_pred_train, y_pred_val are type np.array
    rmse_train, rmse_val are type float
    """
    if scaler.lower().startswith("log_and_divide"):
        divisor = float(scaler.split("_")[-1])
        values_scaled = [divisor * v for v in values]
    elif scaler.lower().startswith("log_and_range"):
        # Split scaler on underscores to extract the min and max values for the range
        elements = scaler.split("_")
        # Calc args for _scale_seq_to_range
        min_value = float(elements[-2])
        max_value = float(elements[-1])
        a = min(train)
        b = max(val)
        # Change name
        args = [a, b, min_value, max_value]
        # may make sense to do this as a for loop (since first 2 will be iteratbvles)
        # and next two are just values
        # Scale the values to values
        values_scaled = [
            _scale_one_value(v, *args)
            if isinstance(v, (int, float))
            else _scale_seq_to_range(v, *args)
            for v in values
        ]
    elif scaler.lower() == "log":
        values_scaled = values
    else:
        raise Exception(
            """Please enter a supported scaling type: log, log_and_divide_a
                        (first take log, then divide by a), or log_and_range_a_b (first take
                        log then scale to range [a, b])."""
        )
    return values_scaled


"""########## PLOT ##########"""


def _plot_actual_vs_pred(y_true, y_pred, rmse=None, repeat=None, name=None, logy=False):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(y_true, "b", label="Test data")
    ax.plot(y_pred, "r", label="Preds")
    ax.legend()

    if rmse is not None and repeat is not None:
        fig_title = f"Actuals vs. Preds - RMSE {rmse:.5f} - Repeat #{repeat}"
        log_title = f"Actuals vs. Preds #{repeat}"
    elif rmse is not None and repeat is None:
        fig_title = f"Actuals vs. Preds - {name} - RMSE {rmse:.5f}"
        log_title = f"Actuals vs. Preds - {name}"
    elif rmse is None and repeat is not None:
        raise Exception("Cannot enter repeat on its own")
    else:
        fig_title = f"Actuals vs. Preds - {name}"
        log_title = fig_title

    ylabel = "BTC Price ($)"
    if logy:
        ylabel = "log(BTC Price USD)"

    ax.set(xlabel="Hours", ylabel=ylabel, title=fig_title)
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
    plt.plot(y_true, "b", label="Actual")
    plt.plot(y_pred, "r", label="Preds")
    plt.legend()
    if len(y_true) > 80000:
        plt.xticks(np.arange(0, 100000, 20000))
    else:
        plt.xticks(np.arange(0, 84000, 14000))
    # Plot predictions for each 10k hours
    for i in range(0, len(y_true) // 10000 + 1):
        if len(y_true) > 80000:
            plt.subplot(2, 5, 2 + i)
        else:
            plt.subplot(2, 4, 2 + i)
        plt.plot(y_true[i * 10000 : (i + 1) * 10000], "b")
        plt.plot(y_pred[i * 10000 : (i + 1) * 10000], "r")
        plt.xticks(
            ticks=np.arange(0, 12000, 2000),
            labels=np.arange(i * 10000, (i + 1) * 10000 + 2000, 2000),
        )
        if i == 1:
            title = f"X_train predictions (broken down) - X_train RMSE {rmse:.5f}"
            plt.title(title)
    plt.tight_layout()
    log_title = "X_train predictions (broken down)"
    wandb.log({log_title: wandb.Image(fig)})
    plt.show()


def _plot_actual_vs_all_preds(y_true, y_preds, rmse_scores):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(y_true, "b", label="Test data")
    for i, y_pred in enumerate(y_preds):
        ax.plot(y_pred, label=f"Preds #{i} - RMSE {rmse_scores[i]:.3f}")
    ax.legend()
    ax.set(xlabel="Hours", ylabel="BTC Price ($)", title="Actuals vs. All Preds")
    wandb.log({f"Actuals vs. All Preds": wandb.Image(fig)})
    plt.show()


def plot_metric(history, metric="loss", ylim=None, start_epoch=0):
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
    title = f"{metric.title()} - Training and Validation"
    ylabel = f"{metric.title()}"
    is_one_minus_metric = False

    if metric.startswith("1-"):
        # i.e. we calculate and plot 1 - metric rather than just metric
        is_one_minus_metric = True
        metric = metric[2:]
    metric = metric.lower()

    fig, ax = plt.subplots()
    num_epochs_trained = len(history.history[metric])
    epochs = range(1, num_epochs_trained + 1)

    values = history.history[metric]
    val_values = history.history[f"val_{metric}"]

    if is_one_minus_metric:
        values = 1 - np.array(history.history[metric])
        val_values = 1 - np.array(history.history[f"val_{metric}"])
    else:
        values = history.history[metric]
        val_values = history.history[f"val_{metric}"]

    ax.plot(epochs[start_epoch:], values[start_epoch:], "b", label="Training")
    ax.plot(epochs[start_epoch:], val_values[start_epoch:], "r", label="Validation")

    ax.set(title=title, xlabel="Epoch", ylabel=ylabel, ylim=ylim)
    ax.legend()
    wandb.log({title: wandb.Image(fig)})
    plt.show()


def plot_train_val_test(train, val, test):
    fig, ax = plt.subplots()
    ax.plot(train, "b", label="Train")
    ax.plot([None for x in train] + [x for x in val], "r", label="Val")
    ax.plot(
        [None for x in train] + [None for x in val] + [x for x in test],
        "g",
        label="Test",
    )
    ax.legend()
    plt.show()


"""########## MEASURE ##########"""
# root mean squared error, or rmse
def _measure_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def measure_rmse_tf(y_true, y_pred):
    m = RootMeanSquaredError()
    m.update_state(y_true, y_pred)
    result = m.result().numpy()
    return result


# Summarize model performance
def summarize_scores(name, scores):
    # print a summary
    scores_m, scores_std = np.mean(scores), np.std(scores)
    ax_title = f"{name}: {scores_m:.3f} RMSE (+/- {scores_std:.3f})"
    print(ax_title)
    log_title = "RMSE Walk-Forward Validation Scores Distribution"
    fig, ax = plt.subplots()
    sns.boxplot(x=scores, ax=ax)
    ax.set(xlabel="RMSE", title=ax_title)
    wandb.log({log_title: wandb.Image(fig)})
    plt.show()


"""########## MODEL BUILD AND FIT ##########"""


def custom_MLP_lr_scheduler(epoch, lr):
    if epoch <= 4:
        return 1e-4
    elif epoch <= 10:
        return 1e-5
    else:
        return 1e-6


def custom_LSTM_lr_scheduler(epoch, lr):
    if epoch <= 3:
        return 1e-3
    # elif epoch <= 17:
    #     return 1e-4
    else:
        return 1e-4


def get_custom_lr_schduler(config):
    """
    Define a custom LR scheduler and return the appropriate one based on
    model_type.

    Note: I cannot do this with one function. The custom_lr_schduler(epoch, lr)
          functions must have a specific form as defined by Keras, so I abstracted
          that away with this func.
    """
    if config.model_type.upper() == "MLP":
        lrs = LearningRateScheduler(custom_MLP_lr_scheduler)
    elif config.model_type.upper() == "LSTM":
        lrs = LearningRateScheduler(custom_LSTM_lr_scheduler)
    else:
        raise Exception("Please enter a supported model_type: MLP or LSTM.")
    return lrs


def get_optimizer(config):
    if config.use_lr_scheduler:
        if config.lr_scheduler == "InverseTimeDecay":
            learning_rate_schedule = InverseTimeDecay(
                config.initial_lr, config.decay_steps, config.decay_rate
            )
        elif config.lr_scheduler == "ExponentialDecay":
            learning_rate_schedule = ExponentialDecay(
                config.initial_lr, config.decay_steps, config.decay_rate
            )
        elif config.lr_scheduler.lower() == "custom":
            if config.optimizer.lower() == "adam":
                optimizer = Adam(learning_rate=config.initial_lr)
            elif config.optimizer.lower() == "rmsprop":
                optimizer = RMSprop(learning_rate=config.initial_lr)
            else:
                raise Exception(
                    """Please enter a supported optimizer: Adam or RMSprop."""
                )
            return optimizer
        else:
            raise Exception(
                """Please enter a supported learning rate scheduler:
                            InverseTimeDecay or ExponentialDecay."""
            )
        if config.optimizer.lower() == "adam":
            optimizer = Adam(learning_rate_schedule)
        elif config.optimizer.lower() == "rmsprop":
            optimizer = RMSprop(learning_rate_schedule)
        else:
            raise Exception("""Please enter a supported optimizer: Adam or RMSprop.""")
    else:
        if config.optimizer.lower() == "adam":
            optimizer = Adam(learning_rate=config.lr)
        elif config.optimizer.lower() == "rmsprop":
            optimizer = RMSprop(learning_rate=config.lr)
        else:
            raise Exception("""Please enter a supported optimizer: Adam or RMSprop.""")
    return optimizer


def build_MLP(config):
    # Do we need to put input_dim=config.n_input in first layer?
    # dense_list = [Dense(config.n_nodes, activation=config.activation) for _ in range(config.num_layers)]
    # dense_list.append(Dense(1))
    # model = Sequential(dense_list)
    model = Sequential(
        [
            Dense(500, activation="relu"),
            Dense(250, activation="relu"),
            Dense(125, activation="relu"),
            Dense(62, activation="relu"),
            Dense(30, activation="relu"),
            Dense(15, activation="relu"),
            Dense(7, activation="relu"),
            Dense(1),
        ]
    )
    optimizer = get_optimizer(config)
    model.compile(
        loss=config.loss, optimizer=optimizer, metrics=[RootMeanSquaredError()]
    )
    return model


def build_LSTM(config):
    # Add (config.num_layers - 1) layers that return sequences
    lstm_list = [
        LSTM(
            config.num_nodes,
            return_sequences=True,
            stateful=True,
            batch_input_shape=(config.n_batch, config.n_input, 1),
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout,
        )
        for _ in range(config.num_layers - 1)
    ]
    # Final layer does not return sequences
    lstm_list.append(
        LSTM(
            config.num_nodes,
            return_sequences=False,
            stateful=True,
            batch_input_shape=(config.n_batch, config.n_input, 1),
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout,
        )
    )
    # Single node output layer
    lstm_list.append(Dense(1))
    model = Sequential(lstm_list)
    optimizer = get_optimizer(config)
    model.compile(
        loss=config.loss, optimizer=optimizer, metrics=[RootMeanSquaredError()]
    )
    return model


def build_LSTM_small(config):
    model = Sequential(
        [
            LSTM(
                100,
                return_sequences=True,
                stateful=True,
                batch_input_shape=(config.n_batch, config.n_input, 1),
            ),
            LSTM(50, return_sequences=True, stateful=True),
            LSTM(25, return_sequences=True, stateful=True),
            LSTM(12, return_sequences=True, stateful=True),
            LSTM(7, stateful=True),
            Dense(1),
        ]
    )
    optimizer = get_optimizer(config)
    model.compile(
        loss=config.loss, optimizer=optimizer, metrics=[RootMeanSquaredError()]
    )
    return model


def build_model(config):
    if config.model_type.upper() == "MLP":
        model = build_MLP(config)
    elif config.model_type.upper() == "LSTM":
        model = build_LSTM(config)
    elif config.model_type.upper() == "LSTM_SMALL":
        model = build_LSTM_small(config)
    else:
        raise Exception("Please enter a supported model type: MLP or LSTM")
    return model


def get_callbacks(config):
    # EarlyStopping
    es = EarlyStopping(
        patience=config.patience,
        restore_best_weights=config.restore_best_weights,
        baseline=config.early_stopping_baseline,
    )
    # WandB
    callbacks_list = [WandbCallback(), es]
    # LearningRateScheduler
    if config.use_lr_scheduler and config.lr_scheduler.lower() == "custom":
        custom_lr_scheduler_callback = get_custom_lr_schduler(config)
        callbacks_list.append(custom_lr_scheduler_callback)
    return callbacks_list


def fit_model(model, config, X_train, X_val, y_train, y_val):
    """
    Fit a DL model and return the history.

    Note that this is model agnostic (MLP vs. LSTM) becuase of our
    data preprocessing. Everything put into fit() is a NumPy array
    and is the correct shape/size such that there will be no errors,
    i.e. for LSTMs the arrays contain n elements where n is a divisor
    of config.n_batch (no excess elements in each batch).
    """
    callbacks_list = get_callbacks(config)

    history = model.fit(
        X_train,
        y_train,
        epochs=config.n_epochs,
        batch_size=config.n_batch,
        verbose=config.verbose,
        shuffle=False,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
    )
    return history


# Define, compile and fit a model on train and val data
def _model_fit(train, config):
    # prepare data
    train_data = _series_to_supervised(train, n_in=config.n_input)
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    # define model
    model = Sequential()
    model.add(
        Dense(config.n_nodes, activation=config.activation, input_dim=config.n_input)
    )
    model.add(Dense(1))
    # compile
    model.compile(
        loss=config.loss, optimizer=config.optimizer, metrics=[RootMeanSquaredError()]
    )
    # fit
    history = model.fit(
        X_train,
        y_train,
        epochs=config.n_epochs,
        batch_size=config.n_batch,
        verbose=config.verbose,
        shuffle=False,
        validation_split=config.val_split,
        callbacks=[WandbCallback()],
    )
    return (model, history)


# forecast with a pre-fit model
def _model_predict(model, history, config):
    # unpack config
    # n_input, _, _, _ = config
    # prepare data
    x_input = np.array(history[-config.n_input :]).reshape(1, config.n_input)
    # forecast, one at a time
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


"""########### EVALUATE ##########"""
# walk-forward validation for univariate data
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
    print(" > %.3f" % error)
    return error, predictions, test


# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=30, plot=True):
    scores = []
    predictions = []
    for i in range(n_repeats):
        print(f"Repeat #{i}")
        score, pred, test = _walk_forward_validation(data, n_test, config)
        scores.append(score)
        predictions.append(pred)
        if plot:
            _plot_actual_vs_pred(test, pred, score, i)
    if plot:
        _plot_actual_vs_all_preds(test, predictions, scores)
    wandb.log({"Repeated walk-forward validation scores": scores})
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


def get_preds(config, model, X_train, X_val, y_train=None, y_val=None):
    """
    Given config, a model and NumPy arrays, calculate and return predictions
    on X_train and X_val.

    For LSTMs, we must fisrt convert the NumPy arrays into tf.data.Dataset
    objects and predict on these.

    For MLPs, we can predict directly on the X arrays.
    If you know you are building an MLP model, you can leave out y_train and
    y_val.
    """
    if config.model_type.upper() == "LSTM":
        # Create train and val tf.data.Datasets
        # Drop excess elements in the final batch
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.batch(config.n_batch, drop_remainder=True)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(config.n_batch, drop_remainder=True)

        y_pred_train = model.predict(train_ds)
        y_pred_val = model.predict(val_ds)
    elif config.model_type.upper() == "MLP":
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
    else:
        raise Exception("Please enter a supported model_type: MLP or LSTM.")
    return y_pred_train, y_pred_val


"""########## WANDB ##########"""


def upload_history_to_wandb(history):
    # Turn into df
    history_df = pd.DataFrame.from_dict(history.history)
    # Turn into wandb Table
    history_table = wandb.Table(dataframe=history_df)
    # Log
    wandb.log({"history": history_table})


"""########## FULL PROCESS ##########"""


def train_and_validate(config):
    # Load data
    train, val = load_train_and_val_data(config)
    # Scale data
    train_scaled, val_scaled = scale_train_val(train, val, scaler=config.scaler)
    # Get data into form Keras needs
    X_train, X_val, y_train, y_val = transform_to_keras_input(
        config, train_scaled, val_scaled, config.n_input
    )
    # Build and fit model
    model = build_model(config)
    history = fit_model(model, config, X_train, X_val, y_train, y_val)
    # Plot loss, rmse, and 1-rmse curves
    plot_metric(history, metric="loss", start_epoch=config.start_plotting_epoch)
    plot_metric(
        history,
        metric="root_mean_squared_error",
        start_epoch=config.start_plotting_epoch,
    )
    plot_metric(
        history,
        metric="1-root_mean_squared_error",
        start_epoch=config.start_plotting_epoch,
    )
    # Store history on wandb
    upload_history_to_wandb(history)
    # Calc predictions
    y_pred_train, y_pred_val = get_preds(config, model, X_train, X_val, y_train, y_val)
    # Convert y_pred_train and y_pred_val into a log scale to enable comparison
    # between different scaling types
    train_log, val_log = scale_train_val(train, val, scaler="log")
    y_pred_train_log, y_pred_val_log = convert_to_log(
        [y_pred_train, y_pred_val], config.scaler, train_log, val_log
    )
    # Create y_train and y_val in log form
    _, _, y_train_log, y_val_log = transform_to_keras_input(
        config, train_log, val_log, config.n_input
    )
    # Calc RMSE between actuals and predictions (both in log scale)
    rmse_train_log = measure_rmse_tf(y_train_log, y_pred_train_log)
    rmse_val_log = measure_rmse_tf(y_val_log, y_pred_val_log)

    # Plot actuals vs. predictions for train and val data (both in log scale)
    _plot_actual_vs_pred(
        y_train_log,
        y_pred_train_log,
        rmse=rmse_train_log,
        name="X_train preds",
        logy=True,
    )
    _plot_actual_vs_pred(
        y_val_log, y_pred_val_log, rmse=rmse_val_log, name="X_val preds", logy=True
    )
    _plot_preds_grid(y_train_log, y_pred_train_log, rmse_train_log)
    return history
