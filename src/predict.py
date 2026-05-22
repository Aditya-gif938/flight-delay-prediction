"""
predict.py
----------
Single-flight prediction interface with risk categorisation.
"""

import numpy as np
import pandas as pd
from model import FlightDelayModel
from feature_engineering import FlightFeatureEngineer


def categorize_risk(prob: float) -> str:
    if prob < 0.30:  return "Low"
    if prob < 0.70:  return "Medium"
    return "High"


def recommendation(risk: str) -> str:
    return {
        "Low":    "Flight is expected on time.",
        "Medium": "Moderate delay risk — monitor updates.",
        "High":   "High delay risk — consider buffer time.",
    }[risk]


class FlightDelayPredictor:
    def __init__(self, model_dir: str = "models"):
        self.model = FlightDelayModel()
        self.model.load(model_dir)
        self.engineer = FlightFeatureEngineer()

    def predict(self, flight_input: dict) -> dict:
        df = pd.DataFrame([flight_input])
        df = self.engineer.transform(df)
        X  = df[self.model.feature_names].values

        prob      = float(self.model.classifier.predict_proba(X)[0, 1])
        will_delay = bool(self.model.classifier.predict(X)[0])
        minutes   = (float(self.model.regressor.predict(X)[0])
                     if will_delay else 0.0)
        risk = categorize_risk(prob)

        return {
            "delay_probability":       round(prob, 3),
            "will_be_delayed":         will_delay,
            "predicted_delay_minutes": round(max(0, minutes), 1),
            "risk_level":              risk,
            "recommendation":          recommendation(risk),
        }


if __name__ == "__main__":
    predictor = FlightDelayPredictor()
    sample = {
        "hour": 8, "day_of_week": 0, "month": 7,
        "route_congestion": 0.7, "temperature": 35.0,
        "visibility": 8.0, "wind_speed": 15.0,
        "precipitation": 2.0, "humidity": 85.0,
        "weather_condition": "Rain",
        "airline": "IndiGo", "origin": "DEL",
        "destination": "BOM", "aircraft_type": "A320",
    }
    result = predictor.predict(sample)
    for k, v in result.items():
        print(f"  {k}: {v}")
