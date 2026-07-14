
# Credit Card Fraud Detection


<p align="center">
  <img src="src\utils\digrams\readme_digrame.png" width="1000">
</p>

## Project Overview

This project aims to detect fraudulent credit card transactions using Machine Learning techniques.

The dataset contains 284,807 transactions with 31 features. Fraudulent transactions represent only 0.172% of the dataset, making it a highly imbalanced classification problem.

To address this issue, SMOTE (Synthetic Minority Oversampling Technique) was applied on the training dataset to balance the classes and improve fraud detection performance.

The project evaluates multiple machine learning models and compares their performance using classification metrics.

---

## Dataset Information

- Total Transactions: 284,807
- Features: 30
- Target Column: Class
- Normal Transactions: 284,315
- Fraudulent Transactions: 492
- Fraud Percentage: 0.172%

---

## Exploratory Data Analysis (EDA)

Key findings from the dataset:

- The dataset is highly imbalanced.
- Fraud cases represent only 0.172% of all transactions.
- No missing values were found.
- Most features are anonymized (V1–V28).
- Transaction Amount contributes to fraud detection.
- The dataset is already normalized except for Time and Amount.
- Correlation between features is generally weak.

---

## Data Preprocessing

The following preprocessing steps were applied:

1. Load Train, Validation, and Test datasets.
2. Separate features and target variable.
3. Apply StandardScaler.
4. Apply SMOTE on the training dataset only.
5. Keep Validation and Test datasets unchanged.
6. Save the scaler for deployment.

---

## Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Voting Classifier

---

## Evaluation Metrics

Since the dataset is highly imbalanced, Accuracy alone is not sufficient.

The following metrics are used:

- Precision
- Recall
- F1-Score
- Classification Report

---

## Project Structure

```text
CreditCardFraudDetection/
│
├── src/
│   ├── Config/
│   │   ├── config.yaml
│   │   └── load.py
│   │
│   ├── data/
│   │   ├── splits/
│   │   │   └── split/
│   │   │       ├── train.csv
│   │   │       ├── val.csv
│   │   │       └── test.csv
│   │   └── load_data.py
│   │
│   ├── Preprocessing/
│   │   ├── preprocessing.py
│   │   └── __init__.py
│   │
│   ├── Modeling/
│   │   ├── Helper/
│   │   │   ├── prepare.py
│   │   │   ├── eval.py
│   │   │   └── save.py
│   │   │
│   │   ├── Train/
│   │   │   ├── logistic_regression.py
│   │   │   ├── random_forest.py
│   │   │   ├── xgboost.py
│   │   │   ├── lightgbm.py
│   │   │   ├── catboost.py
│   │   │   └── voting_classifier.py
│   │   │
│   │   └── model.py
│   │
│   ├── Frontend/
│   ├── Testing/
│   └── utils/
│
├── pkl/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   └── catboost.pkl
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Amany-Ibrahim-1510/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

---

## Training Models

### Logistic Regression

```bash
python src\Modeling\Train\logistic_regression.py
```

### Random Forest

```bash
python src\Modeling\Train\random_forest.py
```

### XGBoost

```bash
python src\Modeling\Train\xgboost.py
```

### LightGBM

```bash
python src\Modeling\Train\lightgbm.py
```

### CatBoost

```bash
python src\Modeling\Train\catboost.py
```

---

## Run Main Model Pipeline

```bash
python src\Modeling\model.py
```

---

## Testing

```bash
python src\Testing\test.py
```

---

## API Deployment

Run FastAPI:

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Frontend

Run Frontend:

```bash
streamlit run src/Frontend/frontend.py
```

Open:

```text
http://localhost:8501
```

---

## Current Result

### Logistic Regression

| Metric | Value |
|----------|----------|
| Accuracy | 97.24% |
| Precision | 4.87% |
| Recall | 88.89% |
| F1 Score | 9.23% |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn
- XGBoost
- LightGBM
- CatBoost
- FastAPI
- Streamlit

---

## Author

Amany Ibrahim Mahmoud

Faculty of Computers and Information

Zagazig University

Artificial Intelligence & Data Analysis