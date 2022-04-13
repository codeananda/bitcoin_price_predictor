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

TIMESTEPS = 168

"""########## LOAD DATA ##########"""


def get_download_and_data_dirs(notebook="local"):
    """
    Return DOWNLOAD_DIR and DATA_DIR depending on the type of notebook being
    used.

    Parameters
    ----------
    notebook : str, optional {'local', 'colab'}
        The type of notebook you are working in

    Returns
    -------
    DOWNLOAD_DIR, DATA_DIR : tuple of Path objects
        Filepaths to the download and data directories respectively
    """
    if notebook.lower() == "colab":
        projects_dir = "/content/drive/MyDrive/1 Projects/"
        DOWNLOAD_DIR = Path(projects_dir + "bitcoin_price_predictor/download")
        DATA_DIR = Path(projects_dir + "bitcoin_price_predictor/data")
    elif notebook.lower() == "local":
        DOWNLOAD_DIR = Path("../download")
        DATA_DIR = Path("../data")
    else:
        raise ValueError(
            f"""You entered {notebook} but the only supported
                notebook types are: colab or local"""
        )
    return DOWNLOAD_DIR, DATA_DIR


def load_dataset_1(notebook="local", drop_first_900_elts=True):
    """Load dataset 1 as defined in data/define_datasets_1_and_2.png

    Parameters
    ----------
    notebook : str, optional {'local', 'colab'}
        The type of notebook you are working in
    drop_first_900_elts : bool, optional
        Whether to drop the first 900 training elements or not. The first 900
        hours (37.5 days) of data contains many missing values. Since it is
        such a small time window, you may want to drop it rather than trying
        to impute missing values.

    Returns
    -------
    train_1, val_1, test_1 : Numpy arrays
        Numpy arrays containing the train/val/test univariate Bitcoin close
        price datasets as defined in data/define_datasets_1_and_2.png
    """
    _, DATA_DIR = get_download_and_data_dirs(notebook)
    with open(DATA_DIR / "train_1.pkl", "rb") as f:
        train_1 = pickle.load(f)
        if drop_first_900_elts:
            train_1 = train_1[900:]

    with open(DATA_DIR / "val_1.pkl", "rb") as f:
        val_1 = pickle.load(f)

    with open(DATA_DIR / "test_1.pkl", "rb") as f:
        test_1 = pickle.load(f)

    return train_1, val_1, test_1


def load_dataset_2(notebook="local"):
    """Load dataset 2 as defined in data/define_datasets_1_and_2.png

    Parameters
    ----------
    notebook : str, optional {'local', 'colab'}
        The type of notebook you are working in

    Returns
    -------
    train_2, val_2 : Numpy arrays
        Numpy arrays containing the train/val univariate Bitcoin close
        price datasets as defined in data/define_datasets_1_and_2.png

        Note: there is no test dataset included due to the limited amount of
        data available after train_2 ends.
    """
    _, DATA_DIR = get_download_and_data_dirs(notebook)
    with open(DATA_DIR / "train_2.pkl", "rb") as f:
        train_2 = pickle.load(f)

    with open(DATA_DIR / "val_2.pkl", "rb") as f:
        val_2 = pickle.load(f)

    return train_2, val_2


def load_train_and_val_data(
    dataset=1, notebook="local", drop_first_900_elts_dataset_1=True
):
    """Convenience function to load just the train and val datasets from
    either dataset 1 or 2 (as defined in data/define_datasets_1_and_2.png)

    Parameters
    ----------
    dataset : int, optional {1, 2}
        The dataset you wish to load as defined in
        data/define_datasets_1_and_2.png
    notebook : str, optional {'local', 'colab'}
        The type of notebook you are working in
    drop_first_900_elts_dataset_1 : bool, optional
        Whether to drop the first 900 training elements of dataset 1 or not.
        The first 900 hours (37.5 days) of data contains many missing values.
        Since it is such a small time window, you may want to drop it rather
        than trying to impute missing values.

    Returns
    -------
    train, val : Numpy arrays
        Numpy arrays containing the train/val univariate Bitcoin close
        datasets as defined in data/define_datasets_1_and_2.png
    """
    if dataset == 1:
        train, val, _ = load_dataset_1(notebook, drop_first_900_elts_dataset_1)
    elif dataset == 2:
        train, val = load_dataset_2(notebook)
    else:
        raise ValueError(
            f"""You entered {dataset} for dataset but the only
            supported values are: 1 or 2"""
        )
    return train, val


"""########## SCALE ##########"""


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

            * 'log' - apply a log transform
            * 'log_and_divide_a' - first apply a log transform, then divide by
               a (a can be any numeric value)
            * 'log_and_range_a_b' - first apply a log transform, then
               scale the dataset to the range [a, b], (must have a < b)

        Note: a and b can be any numeric value e.g. 'log_and_divide_20'
        applies a log transformation, then divides the datasets by 20.

    Returns
    -------
    train_scaled, val_scaled : Tuple of numpy arrays
        Scaled copies of input train and val as dictated by scaler.
    """
    if scaler.lower() == "log":
        train_scaled, val_scaled = _scale_log(train, val)
    elif scaler.lower().startswith("log_and_divide"):
        train_scaled, val_scaled = _scale_log_and_divide(train, val, scaler)
    elif scaler.lower().startswith("log_and_range"):
        train_scaled, val_scaled = _scale_log_and_range(train, val, scaler)
    else:
        raise ValueError(
            """Please enter a supported scaling type: log,
            log_and_divide_a (first take log, then divide by a),
            or log_and_range_a_b (first take log then scale to
            range [a, b])."""
        )
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


def _scale_log_and_divide(train, val, scaler="log_and_divide_20"):
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
    if "log_and_divide" not in scaler:
        raise ValueError(
            f"""scaler must be of the form 'log_and_divide_a' for
                        some number a. You entered {scaler}"""
        )
    # Take log
    train_log, val_log = _scale_log(train, val)
    # The last element of the scaler string the divisor
    divisor = scaler.split("_")[-1]
    # Divide by divisor
    train_log_and_divide = train_log / divisor
    val_log_and_divide = val_log / divisor
    return train_log_and_divide, val_log_and_divide


def _scale_log_and_range(train, val, scaler="log_and_range_0_1"):
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
    if "log_and_range" not in scaler:
        raise ValueError(
            f"""scaler must be of the form 'log_and_range_a_b'
                        for some numbers a and b. You entered {scaler}"""
        )
    # Log scale
    train_log, val_log = _scale_log(train, val)

    # Calculate scaling parameters for _scale_seq_to_range
    scaler_elements = scaler.split("_")
    scaled_min = float(scaler_elements[-2])
    scaled_max = float(scaler_elements[-1])
    if not scaled_min < scaled_max:
        raise ValueError(
            f"""You are trying to scale to the range [a, b] where
                        a = {scaled_min} and b = {scaled_max}. Please choose
                        different values such that a < b."""
        )
    global_min_value = min(min(train_log), min(val_log))
    global_max_value = max(max(train_log), max(val_log))

    scaling_args = [scaled_min, scaled_max, global_min_value, global_max_value]

    train_log_and_range = _scale_seq_to_range(train_log, *scaling_args)
    val_log_and_range = _scale_seq_to_range(val_log, *scaling_args)

    return train_log_and_range, val_log_and_range


def convert_to_log_scale(datasets, scaler="log", log_datasets=None):
    """Convert datasets to a log scale. We assume datasets have previously
    been scaled using scaler, this function undoes the divide or range part of
    scaler and returns a list of log-scaled datasets

    Parameters
    ----------
    datasets : List-like
        Array of datasets that have been scaled with scaler
    scaler: str, optional {'log', 'log_and_divide_a', 'log_and_range_a_b'}
        Scaling that has been applied to datasets. This will be reversed
        and the dataset will be converted to log scale. Options:

            * 'log' - a log transform has been applied, so we return datasets
                      without modification
            * 'log_and_divide_a' - first a log transform, then division by
               a. So, we multiply datasets by a.
            * 'log_and_range_a_b' - first a log transform, then
               scaled to the range [a, b] (must have a < b). So, we scale
               datasets to the min/max values given in log_datasets.

        Note: a and b can be any numeric value e.g. 'log_and_divide_20'
        applies a log transformation, then divides the datasets by 20.
    log_datasets : List-like, optional
        Datasets that are already in a log scale. Only used if scaler is
        'log_and_range_a_b', by default None.

    Raises
    ------
    ValueError
        1. If you enter an unsupported scaling type
        2. If you enter scaler of type 'log_and_range_a_b' without also
           passing log_datasets
    """
    if scaler.lower().startswith("log_and_divide"):
        divisor = float(scaler.split("_")[-1])
        datasets_scaled = [divisor * d for d in datasets]
    elif scaler.lower().startswith("log_and_range"):
        if log_datasets is None:
            raise ValueError(
                f"""You entered scaler {scaler} but have not
                                provided any log_datasets. We need thes to
                                calculate the scaling parameters."""
            )
        # Calculate scaling parameters
        elements = scaler.split("_")
        global_min_value = float(elements[-2])
        global_max_value = float(elements[-1])
        scaled_min = min([min(log_d) for log_d in log_datasets])
        scaled_max = max([max(log_d) for log_d in log_datasets])

        args = [scaled_min, scaled_max, global_min_value, global_max_value]
        datasets_scaled = [_scale_seq_to_range(d, *args) for d in datasets]
    elif scaler.lower() == "log":
        datasets_scaled = datasets
    else:
        raise ValueError(
            f"""You entered {scaler} but the supported scaling
                             types are: log, log_and_divide_a (first take log,
                             then divide by a), or log_and_range_a_b (first
                             take log then scale to range [a, b])."""
        )
    return datasets_scaled


"""########## RESHAPE ##########"""


def _series_to_supervised(univar_time_series, input_seq_length=1, output_seq_length=1):
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
        num_samples = len(univar_time_series) \
                      - input_seq_length \
                      - output_seq_length
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
    array, batch_size=500, timesteps=TIMESTEPS, features=1, array_type="X"
):
    """Transform a numpy array, so that it can be fed into an RNN.

    RNNs require all batches to be the exact same length. This function
    removes excess elements from the array and so ensures all batches are the
    same length.

    Note: this probably isn't the most efficient way to work with the
          tf.data.Dataset API. But keeping everything as numpy arrays makes
          it easier down the road.

    Parameters
    ----------
    array : np.ndarray
        Numpy array containing univariate time-series data
    batch_size : int, optional, default 500
        The number of samples to feed into the RNN on each batch. To keep all
        your data, pick a value that perfectly divides len(array). To keep
        most, pick the value with the smallest remainder after
        len(array) / batch_size
    timesteps : int, optional
        The number of datapoints to feed into the RNN for each sample. If you
        are using 10 datapoints to predict the next one, then set
        timesteps=10, by default 168 i.e. 1 week of hourly data
    features : int
        The number of features the array contains, by default 1 (for
        univariate timeseries data)
    array_type : str {'X', 'y'}
        Whether the array represents X or y data. X data is shaped into
        (num_samples, timesteps, features), y is shaped into
        (num_samples, timesteps), by default 'X'

    Returns
    -------
    array: np.ndarray
        Numpy array shaped so that each batch is the exact same length and is
        ready to be fed into an RNN.
    """
    # Transform to tf.data.Dataset
    a_ds = tf.data.Dataset.from_tensor_slices(array)
    # Put into batches and drop excess elements
    a_batch = a_ds.batch(batch_size, drop_remainder=True)
    # Turn back into a list
    a_list = list(a_batch.as_numpy_iterator())

    # a_numpy is a 3D array shape (total_num_batches, batch_size, timesteps)
    # where total_num_batches = int(len(array) / batch_size)
    a_numpy = np.array(a_list)
    # Reshape into Keras-acceptable shapes
    if array_type.upper() == "X":
        # Reshape to (samples, timesteps, features) if it's an X array
        a_rnn = a_numpy.reshape((-1, timesteps, features))
    elif array_type.lower() == "y":
        # Reshape to (num_samples, timesteps)
        a_rnn = a_numpy.reshape((-1, timesteps))
    else:
        raise ValueError(
            f"""You entered {array_type} but the only supported
                             array_type values are: 'X' and 'y' """
        )
    return a_rnn


def timeseries_to_keras_input(
    train,
    val,
    input_seq_length=TIMESTEPS,
    output_seq_length=1,
    is_rnn=True,
    batch_size=9,
):
    """Given univariate timeseries datasets train and val, transform them into
    a supervised ML problem and (if is_rnn=True) ensure all batches are the
    same length.

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
    train_data = _series_to_supervised(
        train, input_seq_length=input_seq_length, output_seq_length=1
    )
    val_data = _series_to_supervised(
        val, input_seq_length=input_seq_length, output_seq_length=1
    )
    # Create X and y vars (note: batches may not be the same length)
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_val, y_val = val_data[:, :-1], val_data[:, -1]
    # Ensure all batches are the same length if RNN.
    if is_rnn:
        # Transform into RNN-acceptable shapes
        X_train = create_rnn_numpy_batches(
            X_train,
            batch_size=batch_size,
            timesteps=input_seq_length,
            features=1,
            array_type="X",
        )
        X_val = create_rnn_numpy_batches(
            X_val,
            batch_size=batch_size,
            timesteps=input_seq_length,
            features=1,
            array_type="X",
        )
        y_train = create_rnn_numpy_batches(
            y_train,
            batch_size=batch_size,
            timesteps=input_seq_length,
            features=1,
            array_type="y",
        )
        y_val = create_rnn_numpy_batches(
            y_val,
            batch_size=batch_size,
            timesteps=input_seq_length,
            features=1,
            array_type="y",
        )
    return X_train, X_val, y_train, y_val


"""########## MODEL BUILD AND FIT ##########"""


def build_model(
    model_type="LSTM", optimizer="adam", learning_rate=1e-4, loss="mse", **kwargs
):
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
        If you pass a model type that is not one of 'MLP' or 'LSTM'
        (case insensitive)
    """
    if model_type.upper() == "MLP":
        model = build_MLP(optimizer=optimizer, learning_rate=learning_rate, loss=loss)
    elif model_type.upper() == "LSTM":
        model = build_LSTM(
            optimizer=optimizer, learning_rate=learning_rate, loss=loss, **kwargs
        )
    else:
        raise ValueError(
            f"""Supported model types are: MLP or LSTM. You
                             entered {model_type}"""
        )
    return model


def build_MLP(optimizer="adam", learning_rate=1e-4, loss="mse"):
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
    optimizer_object = get_optimizer(optimizer=optimizer, learning_rate=learning_rate)
    model.compile(
        loss=loss, optimizer=optimizer_object, metrics=[RootMeanSquaredError()]
    )
    return model


def build_LSTM(
    optimizer="adam",
    learning_rate=1e-4,
    loss="mse",
    num_nodes=50,
    batch_size=9,
    timesteps=168,
    num_layers=2,
):
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
        The length of each sequence fed into the model, by default 168
        (i.e., one week's worth of hourly data)
    num_layers : int, optional
        The total number of LSTMs to stack, by default 2

    Returns
    -------
    model
        Built and compiled LSTM model with the given params.
    """
    ## BUILD LSTM
    # Add (num_layers - 1) layers that return sequences
    lstm_list = [
        LSTM(
            num_nodes,
            return_sequences=True,
            stateful=True,
            batch_input_shape=(batch_size, timesteps, 1),
        )
        for _ in range(num_layers - 1)
    ]
    # Add a final layer that does not return sequences
    lstm_list.append(
        LSTM(
            num_nodes,
            return_sequences=False,
            stateful=True,
            batch_input_shape=(batch_size, timesteps, 1),
        )
    )
    # Single node output layer
    lstm_list.append(Dense(1))
    model = Sequential(lstm_list)

    ## COMPILE LSTM
    optimizer_object = get_optimizer(optimizer=optimizer, learning_rate=learning_rate)
    model.compile(
        loss=loss, optimizer=optimizer_object, metrics=[RootMeanSquaredError()]
    )
    return model


def get_optimizer(optimizer="adam", learning_rate=1e-4):
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
    if optimizer.lower() == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer.lower() == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(
            f"""You entered {optimizer} but the only supporterd
                             optimizers are: Adam and RMSprop (case
                             insensitive)"""
        )
    return optimizer


def get_callbacks(
    patience=10,
    restore_best_weights=True,
    baseline=None,
    custom_lr_scheduler=None,
    model_type="LSTM",
):
    """Return a list of callbacks containing EarlyStopping, WandB and
    (optionally) a custom learning rate scheduler with the given params

    Parameters
    ----------
    patience : int, optional
        Number of epochs with no improvement after which training will be
        stopped, by default 10
    restore_best_weights : bool, optional
        Whether to restore model weights from the epoch with the best value of
        the monitored quantity. If False, the model weights obtained at the
        last step of training are used, by default True
    baseline : Float, optional
        Baseline value for the monitored quantity. Training will stop if the
        model doesn't show improvement over the baseline, by default None
    custom_lr_scheduler : Bool, optional
        Whether to include a custom learning rate scheduler in the callbacks
        list or not, by default None
    model_type : str, optional
        The type of model the callbacks will be used with, by default 'LSTM'

    Returns
    -------
    List
        A list of callbacks containing EarlyStopping, WandB and
        (optionally) a custom learning rate scheduler with the given params
    """
    # EarlyStopping
    early_stop_cb = EarlyStopping(
        patience=patience, restore_best_weights=restore_best_weights, baseline=baseline
    )
    # WandB
    callbacks_list = [WandbCallback(), early_stop_cb]
    # Custom learning rate scheduler
    if custom_lr_scheduler is not None:
        custom_lr_scheduler_cb = get_custom_lr_scheduler(model_type)
        callbacks_list.append(custom_lr_scheduler_cb)
    return callbacks_list


def get_custom_lr_scheduler(model_type="LSTM"):
    """For the given model_type, return a custom LR scheduling callback

    Parameters
    ----------
    model_type: str, optional {'LSTM', 'MLP'}
        The type of model you want a schedule for, by default 'LSTM'

    Returns
    -------
    lrs: Callback
        Custom learning rate scheduler callback ready for use in training
    """
    if model_type.upper() == "MLP":
        lrs = LearningRateScheduler(custom_MLP_lr_scheduler)
    elif model_type.upper() == "LSTM":
        lrs = LearningRateScheduler(custom_LSTM_lr_scheduler)
    else:
        raise ValueError(
            f"""Supported model types are: MLP or LSTM. You
                             entered {model_type}"""
        )
    return lrs


def custom_MLP_lr_scheduler(epoch, lr):
    """Learning rate schedule for use with MLPs

    Parameters
    ----------
    epoch : int
        The current epoch of training
    lr : float
        The current learning rate

    Returns
    -------
    Float
        The learning rate for the next epoch of training
    """
    if epoch <= 4:
        return 1e-4
    elif epoch <= 10:
        return 1e-5
    else:
        return 1e-6


def custom_LSTM_lr_scheduler(epoch, lr):
    """Learning rate schedule for use with LSTMs

    Parameters
    ----------
    epoch : int
        The current epoch of training
    lr : float
        The current learning rate

    Returns
    -------
    Float
        The learning rate for the next epoch of training
    """
    if epoch <= 3:
        return 1e-3
    # elif epoch <= 17:
    #     return 1e-4
    else:
        return 1e-4


"""########## PLOT ##########"""


def plot_metric(history, metric="loss", ylim=None, start_epoch=0):
    """Plot the given metric from a Keras history

    Parameters
    ----------
    history : tf.keras.callbacks.History
        History object obtained from training
    metric : str, optional
        Metric monitored in training, by default 'loss'
    ylim : tuple of numbers, optional
        Use to set the y-axis limits, by default None
    start_epoch : int, optional
        The epoch at which to start plotting. Useful if plots have large
        values for the first few epochs that render analysis of later epochs
        impossible, by default 0
    """
    # Define plot attrs here as we modify metric later
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
        values = 1 - np.array(values)
        val_values = 1 - np.array(val_values)

    ax.plot(epochs[start_epoch:], values[start_epoch:], "b", label="Training")
    ax.plot(epochs[start_epoch:], val_values[start_epoch:], "r", label="Validation")

    ax.set(title=title, xlabel="Epoch", ylabel=ylabel, ylim=ylim)
    ax.legend()
    wandb.log({title: wandb.Image(fig)})
    plt.show()


def _plot_actual_vs_pred(y_true, y_pred, rmse=None, dataset_name=None, logy=False):
    """Plot y_true and y_pred on the same axis with descriptive titles and
    upload to wandb.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values
    y_pred : np.ndarray
        Array of predicted values
    rmse : float, optional
        The rmse between the two datasets, if given it is included in the
        title for easy manual comparison of plots, by default None
    dataset_name : str, optional
        Name you wish to identify your plots by in WandB and to aid manual
        comparisons. Good choices are 'X_train preds' and 'X_val preds', by
        default None
    logy : bool, optional
        If True the ylabel tells you it is in log scale, by default False
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(y_true, "b", label="Actual")
    ax.plot(y_pred, "r", label="Preds")
    ax.legend()

    fig_title = f"Actuals vs. Preds - {dataset_name}"
    wandb_title = fig_title
    # Add RMSE to title if given
    if rmse is not None:
        fig_title = fig_title + f" - RMSE {rmse:.5f}"

    ylabel = "BTC Price ($)"
    if logy:
        ylabel = "log(BTC Price USD)"

    ax.set(xlabel="Hours", ylabel=ylabel, title=fig_title)
    wandb.log({wandb_title: wandb.Image(fig)})
    plt.show()


def _plot_preds_grid(y_true, y_pred, rmse=None):
    """Plot y_true and y_pred on a 2x4 (or 2x5 depending on size) grid.
    The first subplot shows the entire preds vs. actuals plot. Subsequent
    subplots show 10k timesteps worth of preds vs. actual comparison.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values
    y_pred : np.ndarray
        Array of predicted values
    rmse : float, optional
        The rmse between the two datasets, if given it is included in the
        title for easy manual comparison of plots, by default None
    """
    fig = plt.figure(figsize=(20, 10))
    all_samples_included = len(y_true) > 80000
    # Plot full preds vs. actuals on first axes
    if all_samples_included:
        # Create bigger plot if all samples included
        plt.subplot(2, 5, 1)
    else:
        plt.subplot(2, 4, 1)
    plt.plot(y_true, "b", label="Actual")
    plt.plot(y_pred, "r", label="Preds")
    plt.legend()
    if all_samples_included:
        plt.xticks(np.arange(0, 100000, 20000))
    else:
        plt.xticks(np.arange(0, 84000, 14000))

    # Plot broken down predictions in 10k hour chunks
    chunk = 10000
    for i in range(0, len(y_true) // chunk + 1):
        # Select subplot
        if all_samples_included:
            plt.subplot(2, 5, 2 + i)
        else:
            plt.subplot(2, 4, 2 + i)
        # Plot
        plt.plot(y_true[i * chunk : (i + 1) * chunk], "b")
        plt.plot(y_pred[i * chunk : (i + 1) * chunk], "r")
        # Create xticks and xticklabels
        tick_chunk = 2000
        xticks = np.arange(0, chunk + tick_chunk, tick_chunk)
        xticklabels = np.arange(i * chunk, (i + 1) * chunk + tick_chunk, tick_chunk)
        plt.xticks(ticks=xticks, labels=xticklabels)
    # Figure title
    sup_title = "X_train predictions (broken down)"
    wandb_title = sup_title
    if rmse is not None:
        sup_title = sup_title + f"- RMSE {rmse:.5f}"
    fig.suptitle(sup_title)
    plt.tight_layout()
    wandb.log({wandb_title: wandb.Image(fig)})
    plt.show()


"""########### EVALUATE ##########"""


def calculate_predictions(
    model, X_train, X_val, y_train=None, y_val=None, model_type="LSTM", batch_size=500
):
    """Calculate and return predictions on training and validation data for
    the given model and model_type

    Parameters
    ----------
    model : Keras Model
        Model already fit on data
    X_train : np.ndarray
        Training feature array
    X_val : np.ndarray
        Validation feature array
    y_train : np.ndarray, optional
        Training target array (only needed for model_type='LSTM'), by
        default None
    y_val : np.ndarray, optional
        Validation target array (only needed for model_type='LSTM'), by
        default None
    model_type : str, optional
        The type of model you want to make predictions on, by default 'LSTM'
    batch_size : int, optional
        The number of sequences fed into the model on each batch (only needed
        for model_type='LSTM'), by default 500
    """
    if model_type.upper() == "LSTM":
        # Create train and val tf.data.Datasets and batch to correct size
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(batch_size, drop_remainder=True)

        y_pred_train = model.predict(train_ds)
        y_pred_val = model.predict(val_ds)
    elif model_type.upper() == "MLP":
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
    else:
        raise ValueError(
            f"""You entered {model_type} but the only supported
                            model_types are: MLP or LSTM."""
        )
    return y_pred_train, y_pred_val


def measure_rmse_tf(y_true, y_pred):
    """Calculate the RMSE score between y_true and y_pred and return it.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values
    y_pred : np.ndarray
        Array of predicted values

    Returns
    -------
    rmse : float
        The RMSE between the arrays y_true and y_pred
    """
    m = RootMeanSquaredError()
    m.update_state(y_true, y_pred)
    rmse = m.result().numpy()
    return rmse


"""########## WANDB ##########"""


def upload_history_to_wandb(history):
    """Convenience function to upload a Keras history to W&B

    Parameters
    ----------
    history : tf.keras.callbacks.History
        History object obtained from training
    """
    # Turn into df
    history_df = pd.DataFrame.from_dict(history.history)
    # Turn into wandb Table
    history_table = wandb.Table(dataframe=history_df)
    # Log
    wandb.log({"history": history_table})


"""########## FULL PROCESS ##########"""


def train_and_validate(config):
    # We use hourly close data and want to feed in 1 week of data for each hour of
    # predictions. There are 168 hours in a week
    TIMESTEPS = 168
    BATCH_SIZE = 100
    scaler = "scale_and_range_0_1"
    # Load data
    train, val = load_train_and_val_data(dataset=2, notebook="local")
    # Scale data
    train_scaled, val_scaled = scale_train_val(train, val, scaler=scaler)
    # Get data into form Keras needs
    X_train, X_val, y_train, y_val = timeseries_to_keras_input(
        train_scaled,
        val_scaled,
        input_seq_length=TIMESTEPS,
        output_seq_length=1,
        is_rnn=True,
        batch_size=BATCH_SIZE,
    )
    # Build and fit model
    model = build_model(
        model_type="LSTM", optimizer="adam", learning_rate=1e-4, loss="mse"
    )
    callbacks_list = get_callbacks(
        patience=10,
        restore_best_weights=True,
        custom_lr_scheduler=True,
        model_type="LSTM",
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=500,
        verbose=2,
        shuffle=False,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
    )

    # Plot loss, rmse, and 1-rmse curves
    plot_metric(history, metric="loss", start_epoch=10)
    plot_metric(
        history,
        metric="root_mean_squared_error",
        start_epoch=10,
    )
    plot_metric(
        history,
        metric="1-root_mean_squared_error",
        start_epoch=10,
    )
    # Store history on wandb
    upload_history_to_wandb(history)
    # Calc predictions
    y_pred_train, y_pred_val = calculate_predictions(
        model,
        X_train,
        X_val,
        y_train,
        y_val,
        model_type="LSTM",
        batch_size=BATCH_SIZE,
    )
    # Convert y_pred_train and y_pred_val to log scale to enable comparison
    # between different scaling types
    train_log, val_log = scale_train_val(train, val, scaler="log")
    y_pred_train_log, y_pred_val_log = convert_to_log_scale(
        [y_pred_train, y_pred_val],
        scaler=scaler,
        log_datasets=[train_log, val_log],
    )
    # Create y_train and y_val in log form
    _, _, y_train_log, y_val_log = timeseries_to_keras_input(
        train_log,
        val_log,
        input_seq_length=TIMESTEPS,
        output_seq_length=1,
        is_rnn=True,
        batch_size=BATCH_SIZE,
    )
    # Calc RMSE between actuals and predictions (both in log scale)
    rmse_train_log = measure_rmse_tf(y_train_log, y_pred_train_log)
    rmse_val_log = measure_rmse_tf(y_val_log, y_pred_val_log)

    # Plot actuals vs. predictions for train and val data (both in log scale)
    _plot_actual_vs_pred(
        y_train_log,
        y_pred_train_log,
        rmse=rmse_train_log,
        dataset_name="X_train preds",
        logy=True,
    )
    _plot_actual_vs_pred(
        y_val_log,
        y_pred_val_log,
        rmse=rmse_val_log,
        dataset_name="X_val preds",
        logy=True,
    )
    _plot_preds_grid(y_train_log, y_pred_train_log, rmse_train_log)
    return history
