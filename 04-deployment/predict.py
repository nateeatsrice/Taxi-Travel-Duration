#!/usr/bin/env python
# coding: utf-8

import pickle


from pydantic import BaseModel, Field

from fastapi import FastAPI
import uvicorn

#defining required input datatypes for model
class Distance(BaseModel):
    PU_DO: str
    trip_distance: float = Field(..., ge=0.0)

#defining required datatype output from model
class PredictResponse(BaseModel):
    duration: int

#creating application customer churn prediction
app = FastAPI(title="predict-duration")

#importing pkl file of trained model   
with open('./preprocessor.b','rb') as f_in:
    (booster,dv) = pickle.load(f_in)

# the actual potato and meat 
def predict_duration(ride):
    result = booster.predict_proba(ride)[0, 1]
    return float(result)

@app.post("/predict")

def predict(distance: Distance) -> PredictResponse:
    time = predict_duration(distance.model_dump())

    return PredictResponse(
        duration=time
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9698)