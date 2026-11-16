#Libraries
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

#Model Path
MODEL_PATH = Path("models/model_pipeline.joblib")
model = None

#Uploading model
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError("Not model found. Execute train.py first")
    model = joblib.load(MODEL_PATH)
    yield
    model = None

app = FastAPI(
    title="Home Credit Default Risk API",
    description=("Evaluation with Logistic Regression for Home Credit Risk "),
    version="1.0.0",
    lifespan=lifespan
)

#Structure for receiving data
class CreditApplication(BaseModel):
    AMT_INCOME_TOTAL: float = Field(
        ..., gt=0, json_schema_extra={"example": 150000.0}
    )

    AMT_CREDIT: float = Field(..., gt=0, json_schema_extra={"example": 500000.0})
    AMT_ANNUITY: Optional[float] = Field(
      None, gt=0, json_schema_extra={"example": 25000.0}
  )
    DAYS_BIRTH: int = Field(
        ...,
        lt=0,
        json_schema_extra={"example": -12000},
        description="Negative days since birth",
    )
    DAYS_EMPLOYED: int = Field(..., json_schema_extra={"example": -1500})
    EXT_SOURCE_1: Optional[float] = Field(
        None, ge=0, le=1, json_schema_extra={"example": 0.5}
    )
    EXT_SOURCE_2: Optional[float] = Field(
        None, ge=0, le=1, json_schema_extra={"example": 0.6}
    )
    EXT_SOURCE_3: Optional[float] = Field(
        None, ge=0, le=1, json_schema_extra={"example": 0.4}
    )
    NAME_CONTRACT_TYPE: str = Field(
        ..., json_schema_extra={"example": "Cash loans"}
    )
    CODE_GENDER: str = Field(..., json_schema_extra={"example": "F"})
    FLAG_OWN_CAR: str = Field(..., json_schema_extra={"example": "N"})

#Structure for response
class PredictionResponse(BaseModel):
    default_probability: float
    risk_prediction: int
    risk_level: str

#Checking for server
@app.get("/health")
def health_check():
    return{"status": "ok", "model_loaded": model is not None}

#Main prediction
@app.post("/predict",response_model=PredictionResponse)
def predict_risk(application: CreditApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")

    #Data to DataFrame
    input_data = pd.DataFrame([application.model_dump()])

    #Prediction
    proba = float(model.predict_proba(input_data)[0, 1])
    prediction = int(proba >= 0.5)

    if proba < 0.35:
        risk_level = "Low"

    elif proba < 0.65:
        risk_level = "Medium"

    else:
        risk_level = "High"

    return PredictionResponse(
        default_probability=round(proba,4),
        risk_prediction=prediction,
        risk_level=risk_level

    )

