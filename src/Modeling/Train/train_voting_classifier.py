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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


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

    def try_voting_classifier(self):

        lr_params = config["model"]["voting_classifier"]["params"]["model1"]

        rf_params = config["model"]["voting_classifier"]["params"]["model2"]

        voting_type = config["model"]["voting_classifier"]["params"]["voting"]

        logistic_model = LogisticRegression(
            max_iter=lr_params["max_iter"],
            random_state=lr_params["random_state"]
        )

        random_forest_model = RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            random_state=rf_params["random_state"]
        )

        model = VotingClassifier(
            estimators=[
                ("lr", logistic_model),
                ("rf", random_forest_model)
            ],
            voting=voting_type
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

    model = train.try_voting_classifier()

    save_model(
        model,
        scaler,
        "voting_classifier"
    )

    print("\nSuccessful")
