import pandas as pd
import numpy as np
from typing import Union
import pickle
from fastapi import FastAPI
import uvicorn
import os



def get_user_data(userId):
    data = pd.read_csv("FINAL_NIGHT_DATA.csv")
    #row_data = data[data["userId"]==userId]
    return data

def predict_data(input_row,city,ailment):
    input_row = input_row.drop("Unnamed: 0",axis=1)
    input_row = input_row.drop("userId",axis=1)
    with open('model_final_done.pkl', 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict_proba(input_row)
    hosp_data = pd.read_csv("Xtest_hospdata.csv")
    hosp_data = hosp_data[["Hospital Name","Hospital City","Ailment Grp"]]
    final_val = []
    for i in predictions:
        final_val.append(i[0])


    result = pd.DataFrame()
    result["Hospital Name"] = hosp_data["Hospital Name"]
    result["Hospital City"] = hosp_data["Hospital City"]
    result["Ailment Grp"] = hosp_data["Ailment Grp"]
    result["pct"] = final_val
    
    result = result[(result["Hospital City"]== city)&(result["Ailment Grp"]== ailment)]

    return result


app = FastAPI()

@app.get("/userId/{userId}/city/{city}/disease/{disease}")
def read_item(userId,city,disease):
    row_data = get_user_data(userId)
    b = predict_data(row_data,city,disease)
    return b.to_json(orient='records')

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=os.getenv("PORT", default=3000), log_level="info")

