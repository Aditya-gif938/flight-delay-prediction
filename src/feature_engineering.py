"""
feature_engineering.py
----------------------
Creates 16+ features capturing Indian aviation delay patterns.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


INDIAN_HOLIDAYS = [
    (1, 26), (8, 15), (10, 2),   # Republic, Independence, Gandhi Jayanti
    (11, 14), (12, 25), (1, 1),  # Children's Day, Christmas, New Year
]


class FlightFeatureEngineer:
    """Transforms raw flight + weather data into ML-ready features."""

    def __init__(self):
        self.label_encoders: dict = {}
        self.feature_names: list  = []

    # ── Core transform ───────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Temporal features
        df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
        df["is_holiday"]   = df.apply(
            lambda r: int((r["month"], r["hour"]) in INDIAN_HOLIDAYS), axis=1
        )
        df["is_monsoon"]    = df["month"].isin([6, 7, 8, 9]).astype(int)
        df["is_fog_season"] = df["month"].isin([12, 1, 2]).astype(int)

        # Weather risk composite score
        df["weather_risk"] = self._calc_weather_risk(df)

        # Categorical encodings
        for col in ["airline", "origin", "destination", "aircraft_type", "weather_condition"]:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        self.feature_names = [
            "hour", "day_of_week", "month", "is_weekend", "is_peak_hour",
            "is_holiday", "is_monsoon", "is_fog_season", "weather_risk",
            "temperature", "visibility", "wind_speed", "precipitation",
            "humidity", "route_congestion",
            "airline_enc", "origin_enc", "destination_enc",
            "aircraft_type_enc", "weather_condition_enc",
        ]
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously fitted encoders to new data."""
        df = df.copy()
        df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
        df["is_holiday"]   = 0
        df["is_monsoon"]   = df["month"].isin([6, 7, 8, 9]).astype(int)
        df["is_fog_season"]= df["month"].isin([12, 1, 2]).astype(int)
        df["weather_risk"] = self._calc_weather_risk(df)
        for col in ["airline", "origin", "destination", "aircraft_type", "weather_condition"]:
            le = self.label_encoders.get(col)
            if le:
                df[col + "_enc"] = df[col].apply(
                    lambda v: le.transform([v])[0] if v in le.classes_ else -1
                )
        return df

    # ── Weather risk ─────────────────────────────────────
    @staticmethod
    def _calc_weather_risk(df: pd.DataFrame) -> pd.Series:
        risk = pd.Series(0.0, index=df.index)
        risk += 0.8 * (df["weather_condition"] == "Fog").astype(float)
        risk += 0.3 * (df["weather_condition"] == "Rain").astype(float)
        risk += 0.5 * (df["weather_condition"] == "Thunderstorm").astype(float)
        risk += 0.1 * (df["weather_condition"] == "Haze").astype(float)
        risk += 0.4 * (df["wind_speed"] > 20).astype(float)
        risk += 0.6 * (df["visibility"] < 3).astype(float)
        return risk.round(3)

    def get_feature_names(self) -> list:
        return self.feature_names


if __name__ == "__main__":
    from data_collection import FlightDataCollector
    df_raw = FlightDataCollector(n_samples=1000).collect()
    eng    = FlightFeatureEngineer()
    df_out = eng.fit_transform(df_raw)
    print("Features:", eng.get_feature_names())
    print(df_out[eng.get_feature_names()].head())
