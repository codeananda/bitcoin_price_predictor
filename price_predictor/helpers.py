import pickle
from pathlib import Path


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
        raise Exception('''Set config.notebook to a supported notebook type:
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
        raise Exception('Please enter a supported dataset: 1 or 2')
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
    train, val : Tuple of numpy arrays
        Scaled copies of input train and val as dictated by scaler.
    """
    if scaler.lower() == 'log':
        train, val = _scale_log(train, val)
    elif scaler.lower().startswith('log_and_divide'):
        train, val = _scale_log_and_divide(train, val, scaler)
    elif scaler.lower().startswith('log_and_range'):
        train, val = _scale_log_and_range(train, val, scaler)
    else:
        raise ValueError('''Please enter a supported scaling type: log, log_and_divide_a
                        (first take log, then divide by a), or log_and_range_a_b (first take
                        log then scale to range [a, b]).''')
    return train, val


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