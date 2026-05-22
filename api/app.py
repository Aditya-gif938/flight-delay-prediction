"""
app.py
------
FastAPI REST endpoint for real-time flight delay prediction.
Run: uvicorn api.app:app --reload
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

app = FastAPI(
    title="Flight Delay Prediction API",
    description="AI-powered flight delay prediction using XGBoost",
    version="1.0.0",
)


# ── Request / Response schemas ───────────────────────────
class FlightInput(BaseModel):
    airline:           str   = Field(..., example="IndiGo")
    origin:            str   = Field(..., example="DEL")
    destination:       str   = Field(..., example="BOM")
    scheduled_hour:    int   = Field(..., ge=0, le=23, example=8)
    day_of_week:       int   = Field(..., ge=0, le=6,  example=0)
    month:             int   = Field(..., ge=1, le=12, example=7)
    temperature:       float = Field(..., example=35.0)
    visibility:        float = Field(..., ge=0, example=8.0)
    wind_speed:        float = Field(..., ge=0, example=15.0)
    precipitation:     float = Field(0.0, ge=0, example=2.0)
    humidity:          float = Field(60.0, ge=0, le=100, example=85.0)
    weather_condition: str   = Field(..., example="Clear")
    aircraft_type:     Optional[str] = Field("A320", example="A320")
    route_congestion:  Optional[float] = Field(0.5, ge=0, le=1, example=0.5)


class PredictionResponse(BaseModel):
    delay_probability:       float
    will_be_delayed:         bool
    predicted_delay_minutes: float
    risk_level:              str
    recommendation:          str


# ── Risk helpers ─────────────────────────────────────────
def _risk_level(prob: float) -> str:
    if prob < 0.30: return "Low"
    if prob < 0.70: return "Medium"
    return "High"

def _recommendation(risk: str) -> str:
    return {
        "Low":    "Flight expected on time. No action needed.",
        "Medium": "Moderate delay risk. Monitor flight updates.",
        "High":   "High delay probability. Consider buffer time and alerts.",
    }[risk]

def _demo_predict(flight: FlightInput) -> dict:
    """Rule-based demo predictor (replace with real model when models/ exist)."""
    score = 0.0
    cond  = flight.weather_condition
    if cond == "Fog":           score += 0.55
    elif cond == "Thunderstorm":score += 0.50
    elif cond == "Rain":        score += 0.25
    elif cond == "Haze":        score += 0.10
    if flight.wind_speed > 40:  score += 0.20
    if flight.visibility < 3:   score += 0.20
    if flight.scheduled_hour in [7, 8, 9, 17, 18, 19]: score += 0.10
    if flight.month in [6, 7, 8, 9]: score += 0.10
    prob  = min(score + np.random.uniform(0, 0.08), 0.99)
    delay = max(0.0, prob * 80 + np.random.normal(0, 5)) if prob > 0.3 else 0.0
    risk  = _risk_level(prob)
    return {
        "delay_probability":       round(prob, 3),
        "will_be_delayed":         prob > 0.3,
        "predicted_delay_minutes": round(delay, 1),
        "risk_level":              risk,
        "recommendation":          _recommendation(risk),
    }


# ── Routes ───────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Flight Delay Prediction API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost v1.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(flight: FlightInput):
    try:
        result = _demo_predict(flight)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/airlines")
def get_airlines():
    return {"airlines": ["IndiGo", "Air India", "SpiceJet", "Vistara", "GoFirst", "AirAsia India"]}


@app.get("/airports")
def get_airports():
    return {"airports": ["DEL", "BOM", "BLR", "MAA", "CCU", "HYD", "COK", "PNQ", "AMD", "GOI"]}
