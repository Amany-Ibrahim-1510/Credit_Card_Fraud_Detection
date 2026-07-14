
import pandas as pd
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv(
    r"data\creditcard.csv"
)

# Features & Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train 60% - Temp 40%
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    stratify=y,
    random_state=42
)

# Validation 20% - Test 20%
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

# Save Train
train_df = X_train.copy()
train_df["Class"] = y_train

# Save Validation
val_df = X_val.copy()
val_df["Class"] = y_val

# Save Test
test_df = X_test.copy()
test_df["Class"] = y_test

train_df.to_csv(
    r"data\splits\split\train.csv",
    index=False
)

val_df.to_csv(
    r"data\splits\split\val.csv",
    index=False
)

test_df.to_csv(
    r"data\splits\split\test.csv",
    index=False
)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)