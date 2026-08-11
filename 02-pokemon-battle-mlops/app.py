from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Pokemon Battle Predictor API")
model = joblib.load("pokemon_model.pkl")

class BattleRequest(BaseModel):
    speed_diff: float
    attack_diff: float

@app.get("/")
def home():
    return {"message": "Pokemon Predictor API is running"}

@app.post("/predict")
def predict(data: BattleRequest):
    df = pd.DataFrame([[data.speed_diff, data.attack_diff]], columns=['Speed_Diff', 'Attack_Diff'])
    prediction = model.predict(df)[0]
    return {"winner_predicted": int(prediction)}
