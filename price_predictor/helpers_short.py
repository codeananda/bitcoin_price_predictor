import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf
import pandas as pd


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


def remove_excess_elements(
        array,
        batch_size=1512,
        timesteps=168,
        is_X=False):
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
    a_batch = a_ds.batch(batch_size, drop_remainder=True)
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


def transform_to_keras_input(
        train,
        val,
        n_in,
        model_type='LSTM',
        batch_size=1512,
        timesteps=168):
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
    train_data = _series_to_supervised(train, input_seq_length=n_in)
    val_data = _series_to_supervised(val, input_seq_length=n_in)
    # Create X and y variables
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_val, y_val = val_data[:, :-1], val_data[:, -1]
    if model_type.upper() == 'LSTM':
        # Remove excess elements in the final batch.
        X_train = remove_excess_elements(X_train, batch_size, timesteps,
                                        is_X=True)
        X_val = remove_excess_elements(X_val, batch_size, timesteps,
                                      is_X=True)
        y_train = remove_excess_elements(y_train, batch_size, timesteps)
        y_val = remove_excess_elements(y_val, batch_size, timesteps)
    return X_train, X_val, y_train, y_val


"""########## FULL PROCESS ##########"""
def train_and_validate(config):
    # Load data
    train, val = load_train_and_val_data(config)
    # Scale data
    train_scaled, val_scaled = scale_train_val(train, val, scaler=config.scaler)
    # Get data into form Keras needs
    X_train, X_val, y_train, y_val = transform_to_keras_input(config,
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
    _, _, y_train_log, y_val_log = transform_to_keras_input(
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