import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf
import pandas as pd
import wandb
from wandb.keras import WandbCallback

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# We use hourly close data and want to feed in 1 week of data for each hour of
# predictions. There are 168 hours in a week
TIMESTEPS = 168


"""########## LOAD DATA ##########"""

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
    if config.notebook.lower() == 'colab':
        DOWNLOAD_DIR = Path('/content/drive/MyDrive/1 Projects/bitcoin_price_predictor/download')
        DATA_DIR = Path('/content/drive/MyDrive/1 Projects/bitcoin_price_predictor/data')
    elif config.notebook.lower() == 'local':
        DOWNLOAD_DIR = Path('../download')
        DATA_DIR = Path('../data')
    else:
        raise ValueError('''Set config.notebook to a supported notebook type:
                        colab or local''')
    return DOWNLOAD_DIR, DATA_DIR


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
    with open(DATA_DIR / 'train_1.pkl', 'rb') as f:
        train_1 = pickle.load(f)
        if config.drop_first_900_train_elements:
            train_1 = train_1[900:]

    with open(DATA_DIR / 'val_1.pkl', 'rb') as f:
        val_1 = pickle.load(f)

    with open(DATA_DIR / 'test_1.pkl', 'rb') as f:
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
    with open(DATA_DIR / 'train_2.pkl', 'rb') as f:
        train_2 = pickle.load(f)

    with open(DATA_DIR / 'val_2.pkl', 'rb') as f:
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
        raise ValueError('Set config.dataset to a supported dataset: 1 or 2')
    return train, val


"""########## SCALE ##########"""

def scale_train_val(train, val, scaler='log'):
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
    train_scaled, val_scaled : Tuple of numpy arrays
        Scaled copies of input train and val as dictated by scaler.
    """
    if scaler.lower() == 'log':
        train_scaled, val_scaled = _scale_log(train, val)
    elif scaler.lower().startswith('log_and_divide'):
        train_scaled, val_scaled = _scale_log_and_divide(train, val, scaler)
    elif scaler.lower().startswith('log_and_range'):
        train_scaled, val_scaled = _scale_log_and_range(train, val, scaler)
    else:
        raise ValueError('''Please enter a supported scaling type: log,
                        log_and_divide_a (first take log, then divide by a),
                        or log_and_range_a_b (first take log then scale to
                        range [a, b]).''')
    return train_scaled, val_scaled


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


def _scale_one_value(
        value,
        scaled_min,
        scaled_max,
        global_min,
        global_max
        ):
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


def _scale_seq_to_range(
        seq,
        scaled_min,
        scaled_max,
        global_min=None,
        global_max=None
        ):
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

    scaled_seq = np.array([_scale_one_value(value, scaled_min, scaled_max,
                                                    global_min, global_max) \
                            for value in seq])

    return scaled_seq


def _scale_log_and_divide(train, val, scaler='log_and_divide_20'):
    """First apply a log transform, then divide by the value specified in
    scaler to sequences train and val.

    Parameters
    ----------
    train : np.ndarray
        Training dataset
    val : np.ndarray
        Validation dataset
    scaler: str, optional {'log_and_divide_a'}
        Scaling to apply to train and validation datasets. Options:

            * 'log_and_divide_a' - first apply a log transform, then divide by
               a (a can be any numeric value)

        Note: a can be any numeric value e.g. 'log_and_divide_20'
        applies a log transformation, then divides the datasets by 20.

    Returns
    -------
    train_log_and_divide, val_log_and_divide : Tuple of numpy arrays
        Scaled copies of inputs train and val as dictated by scaler.
    """
    if 'log_and_divide' not in scaler:
        raise ValueError(f"""scaler must be of the form 'log_and_divide_a' for
                        some number a. You entered {scaler}""")
    # Take log
    train_log, val_log = _scale_log(train, val)
    # The last element of the scaler string the divisor
    divisor = scaler.split('_')[-1]
    # Divide by divisor
    train_log_and_divide = train_log / divisor
    val_log_and_divide = val_log / divisor
    return train_log_and_divide, val_log_and_divide


def _scale_log_and_range(train, val, scaler='log_and_range_0_1'):
    """First apply a log transform, then scale train and val sequences to
    the range dicated by scaler. The default ('log_and_range_0_1') scales
    inputs to [0, 1].

    Parameters
    ----------
    train : np.ndarray
        Training dataset
    val : np.ndarray
        Validation dataset
    scaler: str, optional of the form {'log_and_range_a_b'}
        Scaling to apply to train and validation datasets. Options:

            * 'log_and_range_a_b' - first apply a log transform, then
               scale the dataset to the range [a, b]

        Note: a and b can be any numeric values and you must have a < b. So,
        'log_and_range_-5_5' applies a log transformation, then scales the
        datasets to the range [-5, 5].

    Returns
    -------
    train_log_and_divide, val_log_and_divide : Tuple of numpy arrays
        Scaled copies of inputs train and val as dictated by scaler.
    """
    if 'log_and_range' not in scaler:
        raise ValueError(f"""scaler must be of the form 'log_and_range_a_b'
                        for some numbers a and b. You entered {scaler}""")
    # Log scale
    train_log, val_log = _scale_log(train, val)

    # Calculate scaling parameters for _scale_seq_to_range
    scaler_elements = scaler.split('_')
    scaled_min = float(scaler_elements[-2])
    scaled_max = float(scaler_elements[-1])
    if not scaled_min < scaled_max:
        raise ValueError(f'''You are trying to scale to the range [a, b] where
                        a = {scaled_min} and b = {scaled_max}. Please choose
                        different values such that a < b.''')
    global_min_value = min(train_log)
    global_max_value = max(val_log)

    scaling_args = [scaled_min, scaled_max, global_min_value, global_max_value]

    train_log_and_range = _scale_seq_to_range(train_log, *scaling_args)
    val_log_and_range = _scale_seq_to_range(val_log, *scaling_args)

    return train_log_and_range, val_log_and_range


"""########## RESHAPE ##########"""
def _series_to_supervised(
        univar_time_series,
        input_seq_length=1,
        output_seq_length=1):
    """Transform a univariate time-series dataset to a supervised ML problem.
    The number of timesteps in each input sequence is input_seq_length and
    the number of timesteps forcasted is output_seq_length.

    Parameters
    ----------
    univar_time_series : np.ndarray
        Numpy array containing univariate time-series data
    input_seq_length : int, optional
        The number of timesteps for each input sequence i.e. the number of
        timesteps your model will use to make a single prediction, by default
        1
    output_seq_length : int, optional
        The number of timesteps into the future you want to predict, by
        default 1

    Returns
    -------
    np.ndarray
        Numpy array containing the transformed dataset. It has shape
        (num_samples, input_seq_length + output_seq_length)
        num_samples = len(univar_time_series) - input_seq_length - output_seq_length
    """
    df = pd.DataFrame(univar_time_series)
    cols = []
    # Create input sequence cols (t-n, ..., t-1)
    for i in range(input_seq_length, 0, -1):
        cols.append(df.shift(i))
    # Create forecast sequence cols (t, t+1, ..., t+n)
    for i in range(0, output_seq_length):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    agg.dropna(inplace=True)
    return agg.values


def create_rnn_numpy_batches(
        array,
        batch_size=9,
        timesteps=TIMESTEPS,
        is_X=False):
    """Transform a numpy array, so that it can be fed into an RNN.

    RNNs require all batches to be the exact same length. This fucntion
    removes excess elements from the array and so ensures all batches are the
    same length.

    Note: this probably isn't the most efficient way to work with the
          tf.data.Dataset API. But keeping everything as numpy arrays makes
          it easier down the road.

    Parameters
    ----------
    array : np.ndarray
        Numpy array containing univariate time-series data
    batch_size : int, optional, default 9
        The number of samples to feed into the RNN on each batch.
    timesteps : int, optional, default 168 i.e. 1 week of hourly data
        The number of datapoints to feed into the RNN for each sample. If you are
        using 10 datapoints to predict the next one, then set timesteps=10.
    is_X : bool, default False
        Whether the array is an X (feature) array or a y (target) array.
        If is_X=True, the function reshapes the array to
        (samples, timesteps, features). If is_X=False, it returns a 1-D array.

    Returns
    -------
    array: np.ndarray
        Numpy array shaped so that each batch is the exact same length and is
        ready to be fed into an RNN.
    """
    # Transform to tf.data.Dataset
    a_ds = tf.data.Dataset.from_tensor_slices(array)
    # Put into batches and drop excess elements
    a_batch = a_ds.batch(batch_size * timesteps, drop_remainder=True)
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
        a_X_shaped = a_flat.reshape((-1, timesteps, 1))
        return a_X_shaped
    return a_flat


def timeseries_to_keras_input(
        train,
        val,
        input_seq_length=TIMESTEPS,
        output_seq_length=1,
        is_rnn=True,
        batch_size=9):
    """Given univariate timeseries datasets train and val, transform them into a
    supervised ML problem and (if is_rnn=True) ensure all batches are the same
    length.

    Parameters
    ----------
    train : np.ndarray
        Numpy array containing univariate time-series data
    val : np.ndarray
        Numpy array containing univariate time-series data
    input_seq_length : int, optional
        The number of timesteps for each input sequence i.e. the number of
        timesteps your model uses to make a single prediction, by default
        168 (i.e. 1 week's worth of hourly data)
    output_seq_length : int, optional
        The number of timesteps into the future you want to predict, by
        default 1
    is_rnn : bool, optional
        RNNs require all batches to be the same length. If True, the function
        ensures all batches are the same length, if False the last batch may
        be shorter than the others, by default True
    batch_size : int, optional
        The number of samples to feed into the model on each batch. Only
        necessary if is_rnn=True, by default 9

    Returns
    -------
    X_train, X_val, y_train, y_val: Tuple of np.ndarray
        Numpy arrays correctly shaped and batched for Keras MLPs or RNNs.
    """
    # Transform timeseries into a supervised ML problem
    train_data = _series_to_supervised(train,
                                       input_seq_length=input_seq_length,
                                       output_seq_length=1)
    val_data = _series_to_supervised(val,
                                     input_seq_length=input_seq_length,
                                     output_seq_length=1)
    # Create X and y vars (note: batches may not be the same length)
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_val, y_val = val_data[:, :-1], val_data[:, -1]
    # Ensure all batches are the same length if RNN.
    if is_rnn:
        # Remove excess elements in the final batch.
        X_train = create_rnn_numpy_batches(X_train, batch_size, input_seq_length,
                                        is_X=True)
        X_val = create_rnn_numpy_batches(X_val, batch_size, input_seq_length,
                                      is_X=True)
        y_train = create_rnn_numpy_batches(y_train, batch_size, input_seq_length)
        y_val = create_rnn_numpy_batches(y_val, batch_size, input_seq_length)
    return X_train, X_val, y_train, y_val


"""########## MODEL BUILD AND FIT ##########"""


def build_model(model_type='LSTM',
                optimizer='adam',
                learning_rate=1e-4,
                loss='mse',
                **kwargs):
    """Build, compile and return a model of the given type with the given
    params

    Parameters
    ----------
    model_type : str, optional {'LSTM', 'MLP'}
        The type of model to build, by default 'LSTM'
    optimizer : str, optional {'adam', 'rmsprop'}
        The Keras optimizer you would like to use (case insensitive input), by
        default 'adam'
    learning_rate : float, optional
        The learning rate, by default 1e-4
    loss : str, optional {all keras losses are accepted}
        Keras loss to use, by default 'mse'
    **kwargs :
        All other kwargs are passed to the build_LSTM constructor

    Returns
    -------
    model : Keras model
        Compiled Keras model with the given params

    Raises
    ------
    ValueError
        If you pass a model type that is not one of 'MLP' or 'LSTM' (case insensitive)
    """
    if model_type.upper() == 'MLP':
        model = build_MLP(optimizer=optimizer, learning_rate=learning_rate,
                          loss=loss)
    elif model_type.upper() == 'LSTM':
        model = build_LSTM(optimizer=optimizer, learning_rate=learning_rate,
                          loss=loss, **kwargs)
    else:
        raise ValueError(f'''Supported model types are: MLP or LSTM. You
                             entered {model_type}''')
    return model


def build_MLP(optimizer='adam', learning_rate=1e-4, loss='mse'):
    """Build, compile and return an MLP model of with the given params

    Parameters
    ----------
    optimizer : str, optional {'adam', 'rmsprop'}
        The Keras optimizer you would like to use (case insensitive input), by
        default 'adam'
    learning_rate : float, optional
        The learning rate, by default 1e-4
    loss : str, optional {all keras losses are accepted}
        Keras loss to use, by default 'mse'

    Returns
    -------
    model
        MLP sequential Keras model compiled with given params
    """
    model = Sequential([
        Dense(500, activation='relu'),
        Dense(250, activation='relu'),
        Dense(125, activation='relu'),
        Dense(62, activation='relu'),
        Dense(30, activation='relu'),
        Dense(15, activation='relu'),
        Dense(7, activation='relu'),
        Dense(1)
    ])
    optimizer_object = get_optimizer(optimizer=optimizer,
                                    learning_rate=learning_rate)
    model.compile(loss=loss,
                  optimizer=optimizer_object,
                  metrics=[RootMeanSquaredError()])
    return model


def build_LSTM(optimizer='adam', learning_rate=1e-4, loss='mse',
              num_nodes=50, batch_size=9, timesteps=168, num_layers=2):
    """Build, compile and return an LSTM model of with the given params

    Parameters
    ----------
    optimizer : str, optional {'adam', 'rmsprop'}
        The Keras optimizer you would like to use (case insensitive input), by
        default 'adam'
    learning_rate : float, optional
        The learning rate, by default 1e-4
    loss : str, optional {all keras losses are accepted}
        Keras loss to use, by default 'mse'
    num_nodes : int, optional
        The number of nodes in each LSTM layer, by default 50
    batch_size : int, optional
        The number of sequences fed into the LSTM on each batch, by default 9
    timesteps : int, optional
        The length of each sequence fed into the model, by default 168 (i.e.
        one week's worth of hourly data)
    num_layers : int, optional
        The total number of LSTMs to stack, by default 2

    Returns
    -------
    model
        Built and compiled LSTM model with the given params.
    """
    ## BUILD LSTM
    # Add (num_layers - 1) layers that return sequences
    lstm_list = [LSTM(num_nodes,
                      return_sequences=True,
                      stateful=True,
                      batch_input_shape=(batch_size, timesteps, 1)
                      ) for _ in range(num_layers - 1)]
    # Add a final layer that does not return sequences
    lstm_list.append(LSTM(num_nodes,
                      return_sequences=False,
                      stateful=True,
                      batch_input_shape=(batch_size, timesteps, 1)))
    # Single node output layer
    lstm_list.append(Dense(1))
    model = Sequential(lstm_list)

    ## COMPILE LSTM
    optimizer_object = get_optimizer(optimizer=optimizer,
                              learning_rate=learning_rate)
    model.compile(loss=loss,
                  optimizer=optimizer_object,
                  metrics=[RootMeanSquaredError()])
    return model


def get_optimizer(optimizer='adam', learning_rate=1e-4):
    """Given an optimizer and a learning rate, return the optimizer
    object with the learning rate set.

    Parameters
    ----------
    optimizer : str, optional {'adam', 'rmsprop'}
        The Keras optimizer you would like to use (case insensitive input)
    learning_rate : float, optional, default 1e-4
        The learning rate

    Returns
    -------
    optimizer: tf.keras.optimizer object
        Optimizer object with the given learning rate
    """
    if optimizer.lower() == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f'''You entered {optimizer} but the only supporterd
                             optimizers are: Adam and RMSprop (case
                             insensitive)''')
    return optimizer


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
        callbacks=callbacks_list
    )
    return history


def get_callbacks(patience, restore_best_weights, baseline,
                custom_lr_scheduler=None, model_type='LSTM'):
    # EarlyStopping
    early_stop_cb = EarlyStopping(patience=patience,
                                  restore_best_weights=restore_best_weights,
                                  baseline=baseline)
    # WandB
    callbacks_list = [WandbCallback(), early_stop_cb]
    # Custom learning rate scheduler
    if custom_lr_scheduler is not None:
        custom_lr_scheduler_cb = get_custom_lr_schduler(model_type)
        callbacks_list.append(custom_lr_scheduler_cb)
    return callbacks_list


def get_custom_lr_scheduler(model_type='LSTM'):
    """
    Define a custom LR scheduler and return the appropriate one based on
    model_type.

    Note: I cannot do this with one function. The custom_lr_schduler(epoch, lr)
          functions must have a specific form as defined by Keras, so I abstracted
          that away with this func.
    """
    if model_type.upper() == 'MLP':
        lrs = LearningRateScheduler(custom_MLP_lr_scheduler)
    elif model_type.upper() == 'LSTM':
        lrs = LearningRateScheduler(custom_LSTM_lr_scheduler)
    else:
        raise ValueError(f'''Supported model types are: MLP or LSTM. You
                             entered {model_type}''')
    return lrs


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


"""########## FULL PROCESS ##########"""
def train_and_validate(config):
    # Load data
    train, val = load_train_and_val_data(config)
    # Scale data
    train_scaled, val_scaled = scale_train_val(train, val, scaler=config.scaler)
    # Get data into form Keras needs
    X_train, X_val, y_train, y_val = timeseries_to_keras_input(config,
                                                              train_scaled,
                                                              val_scaled,
                                                              config.n_input)
    # Build and fit model
    model = build_model(config)
    history = fit_model(model, config, X_train, X_val, y_train, y_val)
    # Plot loss, rmse, and 1-rmse curves
    plot_metric(history, metric='loss', start_epoch=config.start_plotting_epoch)
    plot_metric(history, metric='root_mean_squared_error', start_epoch=config.start_plotting_epoch)
    plot_metric(history, metric='1-root_mean_squared_error', start_epoch=config.start_plotting_epoch)
    # Store history on wandb
    upload_history_to_wandb(history)
    # Calc predictions
    y_pred_train, y_pred_val = get_preds(config, model, X_train, X_val, y_train, y_val)
    # Convert y_pred_train and y_pred_val into a log scale to enable comparison
    # between different scaling types
    train_log, val_log = scale_train_val(train, val, scaler='log')
    y_pred_train_log, y_pred_val_log = convert_to_log([y_pred_train, y_pred_val],
                                                       config.scaler,
                                                       train_log,
                                                       val_log)
    # Create y_train and y_val in log form
    _, _, y_train_log, y_val_log = timeseries_to_keras_input(
                                                        config,
                                                        train_log,
                                                        val_log,
                                                        config.n_input)
    # Calc RMSE between actuals and predictions (both in log scale)
    rmse_train_log = measure_rmse_tf(y_train_log, y_pred_train_log)
    rmse_val_log = measure_rmse_tf(y_val_log, y_pred_val_log)

    # Plot actuals vs. predictions for train and val data (both in log scale)
    _plot_actual_vs_pred(y_train_log, y_pred_train_log, rmse=rmse_train_log,
                         name='X_train preds', logy=True)
    _plot_actual_vs_pred(y_val_log, y_pred_val_log, rmse=rmse_val_log,
                         name='X_val preds', logy=True)
    _plot_preds_grid(y_train_log, y_pred_train_log, rmse_train_log)
    return history