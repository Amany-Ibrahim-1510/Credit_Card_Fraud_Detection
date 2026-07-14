
"""
load dataframe
load x and t
split train and val
split x_train, x_val, t_train, t_val
"""

from sklearn.model_selection import train_test_split
import pandas as pd

from Config.load import load_config

config = load_config()


def load_df(path):
    if path is None:
        raise ValueError("Path is not defined correctly")

    return pd.read_csv(path)


def load_x_t(df: pd.DataFrame):
    x = df.iloc[:, :-1]
    t = df.iloc[:, -1]
    return x, t


def split_train_val(df: pd.DataFrame, split_sz=0.8):
    sz = int(split_sz * df.shape[0])

    train = df.iloc[:sz, :]
    val = df.iloc[sz:, :]

    return train, val


def split_data(x, t, split_sz=0.2):
    x_train, x_val, t_train, t_val = train_test_split(
        x,
        t,
        test_size=split_sz,
        random_state=config["random_state"],
        stratify=t
    )

    return x_train, x_val, t_train, t_val