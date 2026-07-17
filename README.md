
# Credit Card Fraud Detection


<p align="center">
  <img src="src\utils\digrams\readme_digrame.png" width="1000">
</p>



---

## Project Overview

This project aims to detect fraudulent credit card transactions using Machine Learning techniques.

The dataset contains transactions made by credit cards and is highly imbalanced, where fraudulent transactions represent a very small percentage of the total data.

To handle this imbalance, SMOTE (Synthetic Minority Oversampling Technique) was applied during preprocessing.

Several machine learning models were trained and compared to identify the best-performing model for fraud detection.

---

## Dataset

The dataset contains:

- 284,807 transactions
- 31 features
- Target column: `Class`
  - 0 → Legitimate Transaction
  - 1 → Fraudulent Transaction

### Dataset Characteristics

- Highly imbalanced dataset
- No missing values
- Features V1–V28 are PCA-transformed features
- Amount and Time are original features
- Binary classification problem

---

## Project Structure

```text
CreditCardFraudDetection/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── pkl/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── catboost.pkl
│   └── voting_classifier.pkl
│
├── src/
│   ├── Config/
│   ├── Enum/
│   ├── Helper/
│   ├── Modeling/
│   │   ├── Train/
│   │   └── model.py
│   ├── logs/
│   └── Testing/
│
├── requirements.txt
├── config.yaml
└── README.md
```

---

## Data Preprocessing

The following preprocessing steps were applied:

- Remove duplicate records
- Feature Scaling using StandardScaler
- Handle class imbalance using SMOTE
- Split data into Train and Validation sets

### Before SMOTE

```text
Class 0 : 170579
Class 1 : 305
```

### After SMOTE

```text
Class 0 : 170579
Class 1 : 170579
```

---

## Machine Learning Models

The following models were implemented:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Voting Classifier

---

## Evaluation Metrics

Because the dataset is highly imbalanced, Accuracy alone is not sufficient.

The following metrics were used:

- Precision
- Recall
- F1 Score

The primary metric used for model comparison is **F1 Score**.

---

## Validation Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---------|---------|---------|---------|---------|
| Logistic Regression | 0.9724 | 0.0487 | 0.8889 | 0.0923 |
| Voting Classifier | 0.9895 | 0.1176 | 0.8667 | 0.2072 |
| LightGBM | 0.9958 | 0.2525 | 0.8444 | 0.3887 |
| XGBoost | 0.9974 | 0.3641 | 0.8333 | 0.5068 |
| Random Forest | 0.9981 | 0.4491 | 0.8333 | 0.5837 |
| CatBoost | 0.9991 | 0.5846 | 0.8444 | 0.6909 |

---

## Best Model

Based on validation results:

### CatBoost

```text
Accuracy  : 99.91%
Precision : 58.46%
Recall    : 84.44%
F1 Score  : 69.09%
```

CatBoost achieved the highest F1 Score and was selected as the final model.

---

## Saved Models

The trained models are stored in:

```text
pkl/
```

Available models:

```text
logistic_regression.pkl
random_forest.pkl
xgboost.pkl
lightgbm.pkl
catboost.pkl
voting_classifier.pkl
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/CreditCardFraudDetection.git
```

Move to project directory:

```bash
cd CreditCardFraudDetection
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

## Training

Example:

```bash
python src/Modeling/Train/logistic_regression.py
```

```bash
python src/Modeling/Train/random_forest.py
```

```bash
python src/Modeling/Train/train_xgboost.py
```

```bash
python src/Modeling/Train/train_lightgbm.py
```

```bash
python src/Modeling/Train/train_catboost.py
```

```bash
python src/Modeling/Train/train_voting_classifier.py
```

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
- Pickle

---

## Author

Amany Ibrahim Mahmoud

Faculty of Computers and Information

Zagazig University

Artificial Intelligence Track