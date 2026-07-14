
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# =========================
# Paths
# =========================

TRAIN_PATH = r'D:\zageng\CreditCardFraudDetection\src\data\splits\split\train.csv'
VAL_PATH = r'D:\zageng\CreditCardFraudDetection\src\data\splits\split\trainval.csv'
TEST_PATH = r'D:\zageng\CreditCardFraudDetection\src\data\splits\split\test.api.csv'


# =========================
# Preprocessing Pipeline
# =========================

class PreprocessingPipeline:

    def __init__(self):

        self.scaler = StandardScaler()

    def fit_transform(self, train_df):

        # Features & Target
        X_train = train_df.drop("Class", axis=1)
        y_train = train_df["Class"]

        print("=" * 50)
        print("Before SMOTE")
        print(y_train.value_counts())

        # Scaling
        X_train = self.scaler.fit_transform(X_train)

        # SMOTE
        smote = SMOTE(
            random_state=42,
            k_neighbors=5
        )

        X_train, y_train = smote.fit_resample(
            X_train,
            y_train
        )

        print("=" * 50)
        print("After SMOTE")
        print(pd.Series(y_train).value_counts())

        return X_train, y_train

    def transform(self, df):

        X = df.drop("Class", axis=1)
        y = df["Class"]

        X = self.scaler.transform(X)

        return X, y


# =========================
# Main
# =========================

if __name__ == "__main__":

    print("Loading Data ...")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Train Shape :", train_df.shape)
    print("Val Shape   :", val_df.shape)
    print("Test Shape  :", test_df.shape)

    pipeline = PreprocessingPipeline()

    # Train
    X_train, y_train = pipeline.fit_transform(
        train_df
    )

    # Validation
    X_val, y_val = pipeline.transform(
        val_df
    )

    # Test
    X_test, y_test = pipeline.transform(
        test_df
    )

    print("\nAfter Preprocessing")

    print("X_train :", X_train.shape)
    print("y_train :", y_train.shape)

    print("X_val   :", X_val.shape)
    print("y_val   :", y_val.shape)

    print("X_test  :", X_test.shape)
    print("y_test  :", y_test.shape)

    # Save scaler
    joblib.dump(
        pipeline.scaler,
        "scaler.pkl"
    )

    print("\nScaler Saved Successfully")
        