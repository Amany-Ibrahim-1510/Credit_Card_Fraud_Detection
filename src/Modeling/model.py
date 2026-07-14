import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Modeling.Train.logistic_regression import Train
from Modeling.Helper.prepare import Preparing
from Modeling.Helper.save import save_model


if __name__ == "__main__":

    prepare = Preparing()

    x_train, x_val, t_train, t_val, scaler = (
        prepare.prepare_data()
    )

    print(x_train.shape)
    print(t_train.shape)
    print(x_val.shape)
    print(t_val.shape)

    train = Train(
        x_train,
        x_val,
        t_train,
        t_val
    )

    model = train.try_logistic_regression()

    save_model(
        model,
        scaler,
        "logistic_regression"
    )

    print("Successful")
