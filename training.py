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


def get_Xy_and_model_for_asset(df_proc, asset_id, features_names, params):
    df_proc = df_proc.loc[
        (df_proc[f"Target_{asset_id}"] == df_proc[f"Target_{asset_id}"])
    ]

    # EmbargoCV
    train_test_zip = get_time_series_cross_val_splits(
        df_proc, cv=config.N_FOLD, embargo=3750
    )
    print("entering time series cross validation loop")
    importances = []
    oof_pred = []
    oof_valid = []

    for split, train_test_split in enumerate(train_test_zip):
        gc.collect()

        print(f"doing split {split+1} out of {config.N_FOLD}")
        train_split, test_split = train_test_split
        train_split_index = df_proc["timestamp"].isin(train_split)
        test_split_index = df_proc["timestamp"].isin(test_split)

        train_dataset = lgb.Dataset(
            df_proc.loc[train_split_index, features_names],
            df_proc.loc[train_split_index, f"Target_{asset_id}"].values,
            feature_name=features_names,
        )
        val_dataset = lgb.Dataset(
            df_proc.loc[test_split_index, features_names],
            df_proc.loc[test_split_index, f"Target_{asset_id}"].values,
            feature_name=features_names,
        )

        print(f"number of train data: {len(df_proc.loc[train_split_index])}")
        print(f"number of val data:   {len(df_proc.loc[test_split_index])}")

        model = lgb.train(
            params=params,
            train_set=train_dataset,
            valid_sets=[train_dataset, val_dataset],
            valid_names=["tr", "vl"],
            num_boost_round=5000,
            verbose_eval=100,
            feval=correlation,
        )
        importances.append(model.feature_importance(importance_type="gain"))

        file = f"trained_model_id{asset_id}_fold{split}.pkl"
        pickle.dump(model, open(file, "wb"))
        print(
            f"Trained model was saved to 'trained_model_id{asset_id}_fold{split}.pkl'"
        )
        print("")

        oof_pred += list(model.predict(df_proc.loc[test_split_index, features_names]))
        oof_valid += list(df_proc.loc[test_split_index, f"Target_{asset_id}"].values)

    plot_importance(
        np.array(importances), features_names, plot_top_n=20, figsize=(10, 5)
    )

    return oof_pred, oof_valid
