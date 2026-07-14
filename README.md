# Credit-Card-Fraud-Detection
<p style="text-align:center;">
  <img src='src/utils/digrams/readme_digrame.png' width="1400">
</p>

<h3>

<p>
  <img src='utils/diagrams/ream_me_image.png' style="width:900px; height:500px">
</p>

<h3>
  This project for credit card fraud detection aims to classify fraud transactions using 31 features across 284,807 examples. The data is highly unbalanced (0.172% positive class), so we focus on F1-score and Recall. After trying several algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Decision Tree, SVM, Kernel SVM, and Naive Bayes), we achieved the highest F1-score of 89% with Random Forest, SVM, and Kernel SVM using oversampling.

</h3>

**_From EDA_**

- The Data is Highly imbalanced with only 0.00157% positive class
- All features (except target) are floats
- Most of the features are Anonymous and normalized, 28 features
- the data is very weak (very low correlation)
- there is no missing data (Clean)

```
NYC_TRIP_DURATION/
├── .venv/
├── src/
│   ├── config/
│   ├── data/
│   ├── enums/
│   ├── Frontend/
│   ├── logs/
│   ├── Model/
│   │   ├── Helper/
│   │   ├── Train/
│   │   └── model.py
│   ├── notebook/
│   ├── outputs/
│   ├── Processing/
│   ├── Testing/
│   ├── utils/
│   ├── .gitignore
│   ├── requirements.txt
│   ├── xgboost.pkl
│   └── __init__.py
├── __init__.py
├── main.py
└── README.md

```

## Usage

### 1- Clone Repository
```bash 
# change your_username with your username at github
git clone https://github.com/your_username/NYC_Trip_Duration.git
cd Credit-Card-Fraud-Detection
```

### 2- First check Python Installed
```bash 
python --version
# or
python3 --version
```
- if it doesn't installed install it from [python.org](https://www.python.org/downloads/)

### 3- Check pip installed
```bash
python -m pip --version
```

- if not installed, write 
```bash 
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

python get-pip.py
```

### 4- create Virtual Machine (venv)
``` bash
# Linux/macOS
python3 -m venv venv

# Windows
python -m venv venv
```

### 5- Activate venv
```bash
# Linux/macOS
source venv/bin/activate

# Windows
.\.venv\Scripts\activate
```

### 6- Install Requirments
```bash
cd src 
# NYC_Trip_Duration/src
pip install -r requirements.txt
```

### 7- check Training
```bash
cd src 
# NYC_Trip_Duration/src
python -m Model.model
```

### 8- check testing
```bash
cd src 
# NYC_Trip_Duration/src
python -m Testing.test
```

### 9- run fast-api 
``` bash
# NYC_Trip_Duration
uvicorn main:app --reload
```
- then open ```http://127.0.0.1:8000/docs```
- fast-api screen should show like this 
<p style="text-align:center;">
  <img src='src/utils/digrams/fastapi1.png' width="800">
</p>

- when open \post\predict and try to write test example 
```bash
{
  "vendor_id": 2,
  "pickup_datetime": "2016-06-08 07:36:19",
  "passenger_count": 1,
  "pickup_longitude": -73.985611,
  "pickup_latitude": 40.735943,
  "dropoff_longitude": -73.980331,
  "dropoff_latitude": 40.760468,
  "store_and_fwd_flag": "N"
}
```

- the expected output should show the trip duration prediction
- fast-api screen should show like this 
<p style="text-align:center;">
  <img src='src/utils/digrams/fastapi2.png' width="500" >
</p>

- when open \post\predict_csv and send csv file 
- it expected to send csv file with prediction
<p style="text-align:center;">
  <img src='src/utils/digrams/fastapi3.png' width="500" >
</p>
 

### 10- run frontend
```bash
# NYC_Trip_Duration
streamlit run src/Frontend/frontend.py
```
- open new Terminal and run streemlit 
- then open ``` http://localhost:8501/ ``` 
- predict single example 

<p style="text-align:center;">
  <img src='src/utils/digrams/streemlit1.png' width="400" >
</p>  

- predict csv file 
<p style="text-align:center;">
  <img src='src/utils/digrams/streemlit2.png' width="400" >
</p>

### Metrices:
  We focus on r2-score and mean square error (mse)  
  
### Results:
| Metric            | R2-score (val) | MSE (val) |
|-------------------|----------------|-----------|
| Linear regression | 0.6            | 0.226     |
| Ridge             | 0.59           | 0.232     |
| Neural Network    | 0.586          | 0.234     |
| Xgboost           | 0.76           | 0.134     |


The best model we choose for testing XGBoost that give in val: r2-
score: 0.76% and MSE: 0.134
And for test: r2score: 0.725 and MSE: 0.174

- You can enter [Discord Channel NeuroVision](https://discord.gg/dWFrevZR) for discussion 
- [Ahmed Diab](https://www.linkedin.com/in/ahmed-diab-3b0631245/)
Credit-Card-Fraud-Detection
Credit card fraud detection is the process of identifying and preventing unauthorized or fraudulent use of credit cards. It is a critical aspect of the financial industry, as it helps to safeguard both the cardholder and the card issuer from losses due to fraudulent activity.

How credit card frauds are detected?
Most modern solutions leverage artificial intelligence (AI) and machine learning (ML) to manage data analysis, predictive modeling, decision-making, fraud alerts and remediation activity that occur when individual instances of credit card fraud are detected.



Machine Learning Algorithm Used
In order to detect transactions that are fraud, here we use logistic regression as the transactions occure in binary format (0s and 1s). 0s indicate a normal transaction whereas 1s indicate the fraud credit card detections. Logistic regression works well with binary outcomes that is fraud or not fraud. It helps companies detect the pattern of fraudulent transactions or non-fraudulent transactions.

The image attached below indicates the sigmoid function used in logistic regression for the analysis of fraudulent transaction.

This is an image
