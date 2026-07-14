import os
import sys
import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union

# Add root directory to sys.path to import modules from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helper_fun import load_model
from credit_fraud_utils_data import Processing_Pipeline

app = FastAPI(title="Credit Card Fraud Detection API")

# CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Templates
templates = Jinja2Templates(directory="frontend/templates")

# Global variables for model, threshold, and scaler
model = None
threshold = None
scaler = None

@app.on_event("startup")
async def startup_event():
    global model, threshold, scaler
    try:
        model, threshold, _, scaler = load_model()
        print(f"Model loaded with threshold: {threshold}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Note: If model.pkl doesn't exist, we'll need to train it first.

class Transaction(BaseModel):
    # This matches the expected features for the model
    # The dataset typically has Time, V1-V28, Amount
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

def predict_fraud(df: pd.DataFrame):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Preprocessing
    pipeline = Processing_Pipeline()
    df_processed = pipeline.apply_preprocessing(df)
    
    # Ensure all required columns are present (except the target 'Class')
    # If the model was trained on specific columns, we should filter/order them here
    # Assuming the input has the same columns as the training data
    
    # Scaling
    x_scaled = pipeline.inference_scaling(df_processed, scaler)
    
    # Prediction
    # Check if model has predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_scaled)[:, 1]
        predictions = (probs >= threshold).astype(int)
    else:
        predictions = model.predict(x_scaled)
    
    return predictions, probs if hasattr(model, "predict_proba") else None

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/predict")
async def predict(data: Union[Transaction, List[Transaction]]):
    if isinstance(data, Transaction):
        data = [data]
    
    df = pd.DataFrame([t.dict() for t in data])
    predictions, probs = predict_fraud(df)
    
    results = []
    for i, p in enumerate(predictions):
        result = {"prediction": int(p)}
        if probs is not None:
            result["probability"] = float(probs[i])
        results.append(result)
        
    return results

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Remove 'Class' column if it exists (target column)
        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)
        
        predictions, probs = predict_fraud(df)
        
        df['prediction'] = predictions
        if probs is not None:
            df['probability'] = probs
            
        return df.to_dict(orient='records')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.get("/models")
async def get_models():
    # Assuming the models are saved in the root directory
    models = [f for f in os.listdir('.') if f.endswith('.pkl')]
    return {"models": models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
