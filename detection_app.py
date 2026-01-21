from fastapi import FastAPI
import pandas as pd
from schema import Transaction
from load_model_test import model,business_threshold 
import joblib
scaler=joblib.load("amount_scaler.pkl")
app=FastAPI(title="Fraud_Detection_api",version="1.0.0")

@app.get("/")
def health():
    return{"status":"API is running"}
@app.post("/predict")
def predict(transaction:Transaction):
    X=pd.DataFrame([transaction.dict()])
    X["Amount"]=scaler.transform(X[["Amount"]])
    fraud_prob=float(model.predict_proba(X)[0][1])
    print(fraud_prob)
    if fraud_prob >= business_threshold:
        is_fraud=1
        decision="Fraud detected"
    else:
        is_fraud=0
        decision="allow transaction"
    
    return {"fraud_probability":round(fraud_prob,6),
            "business_threshod":business_threshold,
            "isfraud":is_fraud,
            "decision":decision}


