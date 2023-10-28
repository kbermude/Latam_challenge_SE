from fastapi import FastAPI, HTTPException
from challenge.model import DelayModel
import pandas as pd
from pydantic import BaseModel
import numpy as np

app = FastAPI()
model = DelayModel()
data = pd.read_csv(filepath_or_buffer="data/data.csv")
#Model initialization
features, target = model.preprocess(
            data=data,
            target_column="delay"
        )
model.fit(
    features=features,
    target=target
)
OPERAS = list(set(data.OPERA))

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: dict) -> dict:
    """
    Respond to a POST request at the "/predict" path.       

    Args:
        data (dict): data
    Returns:
        dict
    """
    data = pd.DataFrame(data["flights"],dtype={"OPERA":str,"TIPOVUELO":str,"MES":int})
    if  not data['MES'].between(1, 12,inclusive=True).all():
        raise HTTPException(status_code=404, detail="Incorrect MES.")
    if  np.isin(data["TIPOVUELO"],["N","I"]).all():
        raise HTTPException(status_code=404, detail="Incorrect TIPOVUELO.")
    if  np.isin(data["OPERA"],OPERAS).all():
        raise HTTPException(status_code=404, detail="Incorrect OPERA.")
    results = model.predict(data)    
    return {"predict": results}