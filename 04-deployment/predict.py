#!/usr/bin/env python
# coding: utf-8

import pickle

from pydantic import BaseModel, Field

from fastapi import FastAPI
import uvicorn

#defining required input datatypes for model
class Request(BaseModel):
    PU_DO: str
    trip_distance: float = Field(..., ge=0.0)

#defining required datatype output from model
class Response(BaseModel):
    duration_prediction: float

#creating application customer churn prediction
app = FastAPI(title="predict-duration")

#importing pkl file of trained model   
with open('./preprocessor.b','rb') as f_in:
    (dv,booster) = pickle.load(f_in)

# the actual potato and meat 
def predict_duration(ride, dv, booster) -> float:
    X = dv.transform([ride])
    duration = booster.predict(X)
    return float(duration)

@app.post("/predict")

def predict(distance:Request) -> Response:
    time = predict_duration(distance.model_dump(),dv=dv,booster=booster)

    return Response(
        duration_prediction=time
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9698)