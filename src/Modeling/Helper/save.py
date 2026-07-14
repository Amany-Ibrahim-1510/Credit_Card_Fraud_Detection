import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle


def save_model(model, scaler, name="model"):

    model_dict = {
        "model": model,
        "scaler": scaler
    }

    os.makedirs("pkl", exist_ok=True)

    filename = f"pkl/{name}.pkl"

    with open(filename, "wb") as file:
        pickle.dump(model_dict, file)

    print(f"{name}.pkl saved successfully")
