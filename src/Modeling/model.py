import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

from Helper.prepare import Preparing
from Helper.save import save_model

from Train.train_catboost import Train


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

    print("Training Best Model (CatBoost)...")

    model = train.try_catboost()

    save_model(
        model,
        scaler,
        "best_model"
    )

    print("\nBest model saved successfully!")