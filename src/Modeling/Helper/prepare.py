import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.append(ROOT_DIR)


import pandas as pd

from Preprocessing.preprocessing import PreprocessingPipeline


class Preparing:

    def __init__(self):

        self.pipeline = PreprocessingPipeline()

        self.train_df = pd.read_csv(
            r"src\data\splits\split\train.csv"
        )

        self.val_df = pd.read_csv(
            r"src\data\splits\split\val.csv"
        )

    def prepare_data(self):

        X_train, y_train = self.pipeline.fit_transform(
            self.train_df
        )

        X_val, y_val = self.pipeline.transform(
            self.val_df
        )

        return (
            X_train,
            X_val,
            y_train,
            y_val,
            self.pipeline.scaler
        )


if __name__ == "__main__":

    prepare = Preparing()

    X_train, X_val, y_train, y_val, scaler = (
        prepare.prepare_data()
    )

    print(X_train.shape)
    print(y_train.shape)

    print(X_val.shape)
    print(y_val.shape)