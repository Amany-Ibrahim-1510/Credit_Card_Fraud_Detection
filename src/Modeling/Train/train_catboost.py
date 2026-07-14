
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

from catboost import CatBoostClassifier


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


    def try_catboost(self):

        params = config["model"]["catboost"]["params"]

        model = CatBoostClassifier(
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            iterations=params["iterations"],
            random_seed=params["random_seed"],
            verbose=0
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


    model = train.try_catboost()


    save_model(
        model,
        scaler,
        "catboost"
    )


    print("\nSuccessful")