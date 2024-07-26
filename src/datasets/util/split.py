import pandas as pd

import torch_frame
import torch
import numpy as np
import itertools


def apply_split(df: pd.DataFrame, split_type: str, splits: list, timestamp_col: str) -> pd.DataFrame:
    if split_type == 'temporal_daily':
        df = temporal_balanced_split(df, splits, timestamp_col)
    elif split_type == 'temporal':
        df = temporal_split(df, splits, timestamp_col)
    else:
        df = random_split(df, splits)
    return df


def random_split(df: pd.DataFrame, splits: list) -> pd.DataFrame:
    df['split'] = torch_frame.utils.generate_random_split(length=len(df), seed=0, train_ratio=splits[0], val_ratio=splits[1])
    return df


def temporal_split(df: pd.DataFrame, splits: list, timestamp_col: str) -> pd.DataFrame:
    assert timestamp_col in df.columns, \
        f'Split is only available for datasets with a {timestamp_col} column'
    # create a mask column that stores the sorted indices based on the timestamp column
    mask = df[timestamp_col].argsort() - 1

    train_size = int(df.shape[0] * splits[0])
    validation_size = int(df.shape[0] * splits[1])

    # if mask < train_size, then it is train, if mask < train_size + validation_size, then it is validation, else test
    df['split'] = 2
    df.loc[mask < train_size, 'split'] = 0
    df.loc[(mask >= train_size) & (mask < train_size + validation_size), 'split'] = 1

    return df


def temporal_balanced_split(df: pd.DataFrame, splits: list, timestamp_col: str) -> pd.DataFrame:
    assert timestamp_col in df.columns, \
        f"Split is only available for datasets with a {timestamp_col} column."
    # print example timestamps
    df[timestamp_col] = df[timestamp_col] - df[timestamp_col].min()

    timestamps = torch.Tensor(df[timestamp_col].to_numpy())
    n_days = int(timestamps.max() / (3600 * 24) + 1)

    daily_inds, daily_trans = [], []  # irs = illicit ratios, inds = indices, trans = transactions
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])

    split_per = splits
    daily_totals = np.array(daily_trans)
    d_ts = daily_totals
    I = list(range(len(d_ts)))
    split_scores = dict()
    for i, j in itertools.combinations(I, 2):
        if j >= i:
            split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            # split_totals = [d_ts[:i].sum(), d_ts[i:].sum()]
            split_totals_sum = np.sum(split_totals)
            split_props = [v / split_totals_sum for v in split_totals]
            split_error = [abs(v - t) / t for v, t in zip(split_props, split_per)]
            score = max(split_error)  # - (split_totals_sum/total) + 1
            split_scores[(i, j)] = score
        else:
            continue

    i, j = min(split_scores, key=split_scores.get)
    # split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
    split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]

    # Now, we seperate the transactions based on their indices in the timestamp array
    split_inds = {k: [] for k in range(3)}
    for i in range(3):
        for day in split[i]:
            split_inds[i].append(daily_inds[
                                     day])  # split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately
    # According to Efe, the above code might break on Amazon Fashion, so try this instead:
    # split_inds = {k: [] for k in range(3)}
    #     for i in range(3):
    #         for day in split[i]:
    #             if daily_inds[day].numel() > 0:
    #                 # print(daily_inds[day])
    #                 temp_tensor = daily_inds[day].unsqueeze(0)
    #                 split_inds[i].extend(temp_tensor)  # split_inds contains a list for each split (tr, val, te) which
    #                 # contains the indices of each day separately

    # tr_inds = torch.cat(split_inds[0])
    val_inds = torch.cat(split_inds[1])
    te_inds = torch.cat(split_inds[2])

    # add a new split column to df
    df['split'] = 0

    # Set values for val_inds and te_inds
    df.loc[val_inds, 'split'] = 1
    df.loc[te_inds, 'split'] = 2
    return df
