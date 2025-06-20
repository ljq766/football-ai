from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import sqlite3
from pydantic import BaseModel
from typing import List

app = FastAPI()

# 加载模型和特征列
model = joblib.load('prediction_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# 定义输入数据格式
class MatchInput(BaseModel):
    home_team: str
    away_team: str

@app.post("/predict")
def predict(input: MatchInput):
    # 构建特征向量
    input_data = pd.DataFrame([[input.home_team, input.away_team]], 
                             columns=['HomeTeam', 'AwayTeam'])
    input_data = pd.get_dummies(input_data)
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    prediction = model.predict_proba(input_data[feature_columns])[0][1]
    result = "主队胜" if prediction > 0.5 else "客队不败"
    return {"result": result, "confidence": prediction}

@app.get("/history")
def get_history():
    conn = sqlite3.connect('football_db.sqlite')
    history_df = pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY match_date DESC LIMIT 10", conn)
    conn.close()
    return history_df.to_dict(orient="records")

@app.get("/accuracy")
def get_accuracy():
    conn = sqlite3.connect('football_db.sqlite')
    history_df = pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY match_date DESC LIMIT 10", conn)
    conn.close()
    accuracy = history_df['correct'].mean() if not history_df.empty else 0
    return {"accuracy": accuracy}

@app.get("/")
def root():
    return {"message": "足球赛事AI预测API (FastAPI版) 已启动"}
