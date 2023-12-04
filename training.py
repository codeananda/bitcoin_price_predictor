import gc
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

import config


# For CV score calculation
def corr_score(pred, valid):
    pred = np.array(pred)
    valid = np.array(valid)
    len_data = len(pred)
    mean_pred = pred.mean()
    mean_valid = valid.mean()
    var_pred = pred.var(dtype=np.float32)
    var_valid = valid.var(dtype=np.float32)

    cov = np.sum((pred * valid)) / len_data - mean_pred * mean_valid
    corr = cov / np.sqrt(var_pred * var_valid)

    return corr


# Define evaluation metric for LightGBM
def correlation(a, train_data):
    b = train_data.get_label()

    a = np.ravel(a)
    b = np.ravel(b)

    corr = corr_score(a, b)

    return "corr", corr, True


# For CV score calculation
def wcorr_score(pred, valid, weight):
    len_data = len(pred)
    sum_w = np.sum(weight)
    mean_pred = np.sum(pred * weight) / sum_w
    mean_valid = np.sum(valid * weight) / sum_w
    var_pred = np.sum(weight * np.square(pred - mean_pred)) / sum_w
    var_valid = np.sum(weight * np.square(valid - mean_valid)) / sum_w

    cov = np.sum((pred * valid * weight)) / sum_w - mean_pred * mean_valid
    corr = cov / np.sqrt(var_pred * var_valid)

    return corr


# from: https://blog.amedama.jp/entry/lightgbm-cv-feature-importance
# (used in nyanp's Optiver solution)
def plot_importance(importances, features_names, plot_top_n=20, figsize=(10, 10)):
    importance_df = pd.DataFrame(data=importances, columns=features_names)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:plot_top_n]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale("log")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Importance")
    sns.boxplot(data=sorted_importance_df[plot_cols], orient="h", ax=ax)
    plt.show()


def embargo_cv(df, n_splits=config.N_FOLD, embargo_period=336):
    """
    Perform embargo cross-validation on time series data.

    This function uses TimeSeriesSplit for creating cross-validation splits and
    then applies an embargo period to ensure that the test set is separated from
    the training set by a specified time gap.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing time series data.
    n_splits : int
        Number of splits for cross-validation.
    embargo_period : int
        The number of time units to embargo data after the training set.
        Two weeks of hourly data (14 x 24 = 336) by default.

    Yields
    ------
    train_index : np.array
        Indices for training data in each split.
    test_index : np.array
        Indices for testing data in each split.

    Example
    -------
    >>> df = pd.DataFrame(...)  # your time series DataFrame
    >>> n_splits = 5  # number of splits for cross-validation
    >>> embargo_period = 10  # embargo period in time units (e.g., days)
    >>> for train_index, test_index in embargo_cv(df, n_splits, embargo_period):
    ...     train_data = df.iloc[train_index]
    ...     test_data = df.iloc[test_index]
    ...     # Model training and evaluation goes here
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_index, test_index in tscv.split(df):
        # Apply the embargo period
        max_train_index = train_index.max()
        test_index = test_index[test_index > max_train_index + embargo_period]

        yield train_index, test_index


def train_and_evaluate_time_series_model(
    df_proc: pd.DataFrame, feature_names: list[str], params: dict
):
    """
    Trains a LightGBM model for a specified asset using time series cross-validation,
    and returns out-of-fold predictions and true values.

    Parameters
    ----------
    df_proc : DataFrame
        The preprocessed DataFrame.
    feature_names : list of str
        The names of the features used in the model.
    params : dict
        The parameters for the LightGBM model.

    Returns
    -------
    tuple of (list, list)
        A tuple containing two lists:
        - Out-of-fold predictions.
        - Corresponding true values.
    """
    print("Entering time series cross-validation loop")

    importances = []
    oof_pred = []
    oof_valid = []

    for fold, (train_split, test_split) in enumerate(
        embargo_cv(df_proc, n_splits=config.N_FOLD, embargo_period=336)
    ):
        gc.collect()
        print(f"\nProcessing split {fold + 1}/{config.N_FOLD}")

        train_split_index = df_proc.index[train_split]
        test_split_index = df_proc.index[test_split]

        train_dataset = lgb.Dataset(
            df_proc.loc[train_split_index, feature_names],
            label=df_proc.loc[train_split_index, "target"].values,
            feature_name=feature_names,
        )
        val_dataset = lgb.Dataset(
            df_proc.loc[test_split_index, feature_names],
            label=df_proc.loc[test_split_index, "target"].values,
            feature_name=feature_names,
        )

        print(f"Number of train data: {len(train_split_index)}")
        print(f"Number of val data: {len(test_split_index)}")

        model = lgb.train(
            params=params,
            train_set=train_dataset,
            valid_sets=[train_dataset, val_dataset],
            valid_names=["train", "val"],
            num_boost_round=5000,
            feval=correlation,
        )
        importances.append(model.feature_importance(importance_type="gain"))

        file_name = f"trained_model_id_fold{fold}.pkl"
        pickle.dump(model, open(file_name, "wb"))
        print(f"Trained model saved to '{file_name}'")

        y_pred = model.predict(df_proc.loc[test_split_index, feature_names])
        y_val = df_proc.loc[test_split_index, "target"].values

        oof_pred.extend(y_pred)
        oof_valid.extend(y_val)

    plot_importance(
        np.array(importances), feature_names, plot_top_n=20, figsize=(10, 5)
    )

    return oof_pred, oof_valid


class BaselinePreviousHour:
    """Prediction = previous hour's close"""

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X["close"].values

    def run_embargo_cv(self, df_proc: pd.DataFrame):
        """
        Runs embargo cross-validation for the baseline model.

        Parameters
        ----------
        df_proc : DataFrame
            The preprocessed DataFrame.

        Returns
        -------
        tuple of (list, list)
            A tuple containing two lists:
            - Out-of-fold predictions.
            - Corresponding true values.
        """
        oof_pred = []
        oof_valid = []

        for fold, (train_split, test_split) in enumerate(
            embargo_cv(df_proc, n_splits=config.N_FOLD, embargo_period=336)
        ):
            gc.collect()
            print(f"\nProcessing split {fold + 1}/{config.N_FOLD}")

            train_split_index = df_proc.index[train_split]
            test_split_index = df_proc.index[test_split]

            y_pred = self.predict(df_proc.loc[test_split_index])
            y_val = df_proc.loc[test_split_index, "target"].values

            oof_pred.extend(y_pred)
            oof_valid.extend(y_val)

        return oof_pred, oof_valid
