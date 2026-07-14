import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../..'
        )
    )
)

from Config.load import load_config

from Modeling.Helper.prepare import Preparing
from Modeling.Helper.eval import eval_model
from Modeling.Helper.save import save_model

from xgboost import XGBClassifier


config = load_config()


class Train():

    def __init__(
        self,
        x_train,
        x_val,
        t_train,
        t_val
    ):

        self.x_train = x_train
        self.x_val = x_val

        self.t_train = t_train
        self.t_val = t_val

    def try_xgboost(self):

        params = config["model"]["xgboost"]["params"]

        model = XGBClassifier(
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            random_state=params["random_state"],
            eval_metric="logloss"
        )

        model.fit(
            self.x_train,
            self.t_train
        )

        print("\nTrain Results")

        eval_model(
            model,
            self.x_train,
            self.t_train,
            "train"
        )

        print("\nValidation Results")

        eval_model(
            model,
            self.x_val,
            self.t_val,
            "val"
        )

        return model


if __name__ == "__main__":

    prepare = Preparing()

    x_train, x_val, t_train, t_val, scaler = (
        prepare.prepare_data()
    )

    train = Train(
        x_train,
        x_val,
        t_train,
        t_val
    )

    model = train.try_xgboost()

    save_model(
        model,
        scaler,
        "xgboost"
    )

    print("\nSuccessful")