import gc
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import config


# For CV score calculation
def corr_score(pred, valid):
    len_data = len(pred)
    mean_pred = np.sum(pred) / len_data
    mean_valid = np.sum(valid) / len_data
    var_pred = np.sum(np.square(pred - mean_pred)) / len_data
    var_valid = np.sum(np.square(valid - mean_valid)) / len_data

    cov = np.sum((pred * valid)) / len_data - mean_pred * mean_valid
    corr = cov / np.sqrt(var_pred * var_valid)

    return corr


# TODO - is this needed?
# define the evaluation metric
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


# from: https://www.kaggle.com/code/nrcjea001/lgbm-embargocv-weightedpearson-lagtarget/
def get_time_series_cross_val_splits(data, cv=config.N_FOLD, embargo=3750):
    all_train_timestamps = data["timestamp"].unique()
    len_split = len(all_train_timestamps) // cv
    test_splits = [
        all_train_timestamps[i * len_split : (i + 1) * len_split] for i in range(cv)
    ]
    # fix the last test split to have all the last timestamps, in case the number of timestamps wasn't divisible by cv
    rem = len(all_train_timestamps) - len_split * cv
    if rem > 0:
        test_splits[-1] = np.append(test_splits[-1], all_train_timestamps[-rem:])

    train_splits = []
    for test_split in test_splits:
        test_split_max = int(np.max(test_split))
        test_split_min = int(np.min(test_split))
        # get all of the timestamps that aren't in the test split
        train_split_not_embargoed = [
            e
            for e in all_train_timestamps
            if not (test_split_min <= int(e) <= test_split_max)
        ]
        # embargo the train split so we have no leakage. Note timestamps are expressed in seconds, so multiply by 60
        embargo_sec = 60 * embargo
        train_split = [
            e
            for e in train_split_not_embargoed
            if abs(int(e) - test_split_max) > embargo_sec
            and abs(int(e) - test_split_min) > embargo_sec
        ]
        train_splits.append(train_split)

    # convenient way to iterate over train and test splits
    train_test_zip = zip(train_splits, test_splits)
    return train_test_zip


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
