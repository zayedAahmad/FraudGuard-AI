from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(title="Banking Fraud Detection API", version="1.0")


MODEL = joblib.load("models/fraud_model.pkl")
FEATURES = joblib.load("models/feature_names.pkl")


class Transaction(BaseModel):
    data: dict


@app.get("/")
def home():
    return {"message": "Fraud Detection System is Online"}


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        df = pd.DataFrame([transaction.data])

        # Keep only the expected features in the correct order
        df = df[FEATURES]

        probability = MODEL.predict_proba(df)[:, 1][0]
        is_fraud = probability >= 0.25

        if probability >= 0.80:
            risk_level = "High"
        elif probability >= 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "is_fraud": bool(is_fraud),
            "fraud_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "recommendation": "BLOCK TRANSACTION" if is_fraud else "ALLOW TRANSACTION",
            "status": "Verified"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)