from fastapi import FastAPI, HTTPException
from challenge.model import DelayModel
import pandas as pd
from pydantic import BaseModel
import numpy as np

app = FastAPI()
model = DelayModel()
datafull = pd.read_csv(filepath_or_buffer="data/data.csv", low_memory=False)
concat_df = pd.concat([
            pd.get_dummies(datafull['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(datafull['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(datafull['MES'], prefix = 'MES')], 
            axis = 1
        )
cols = list(concat_df.columns)
lencols = len(cols)

features, target = model.preprocess(
            data=datafull,
            target_column="delay"
        )
model.fit(
    features=features,
    target=target
)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.get("/")
def home():
    return {"message":"api iniciada"}

@app.post("/predict", status_code=200)
async def post_predict(data: dict) -> dict:
    df = pd.DataFrame(data["flights"])
    rows = []
    for _,row in df.iterrows():
        f_row = [0]*lencols
        opera = f"OPERA_{row['OPERA']}"
        tipovuelo = f"TIPOVUELO_{row['TIPOVUELO']}"
        mes = f"MES_{row['MES']}"
        if opera not in cols:
            raise HTTPException(status_code=400, detail="OPERA incorrecto.")
        f_row[cols.index(opera)]=1
        if tipovuelo not in cols:
            raise HTTPException(status_code=400, detail="TIPOVUELO incorrecto.")
        f_row[cols.index(tipovuelo)]=1
        if mes not in cols:
            raise HTTPException(status_code=400, detail="MES incorrecto.")
        f_row[cols.index(mes)]=1
        rows.append(f_row)
    df= pd.DataFrame(rows,columns=cols)[model.topfeatures]
    results = model.predict(df)
    return {"predict": results}