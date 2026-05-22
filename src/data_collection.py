"""
data_collection.py
------------------
Generates realistic synthetic flight + weather data for the
Indian aviation context (monsoon, fog, festival seasons).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ── Constants ────────────────────────────────────────────
AIRLINES = ["IndiGo", "Air India", "SpiceJet", "Vistara", "GoFirst", "AirAsia India"]
AIRPORTS = ["DEL", "BOM", "BLR", "MAA", "CCU", "HYD", "COK", "PNQ", "AMD", "GOI"]
WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Fog", "Thunderstorm", "Haze"]
AIRCRAFT_TYPES = ["A320", "B737", "A321", "B777", "ATR72", "Q400"]

ROUTE_PAIRS = [
    ("DEL", "BOM"), ("DEL", "BLR"), ("DEL", "MAA"), ("DEL", "HYD"),
    ("BOM", "BLR"), ("BOM", "COK"), ("BOM", "CCU"), ("BOM", "PNQ"),
    ("BLR", "MAA"), ("BLR", "HYD"), ("CCU", "DEL"), ("HYD", "MAA"),
]


class FlightDataCollector:
    """Generates synthetic flight & weather data for the Indian aviation context."""

    def __init__(self, n_samples: int = 5000, random_state: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(random_state)

    # ── Date / Time ──────────────────────────────────────
    def _generate_dates(self) -> pd.DatetimeIndex:
        start = datetime(2022, 1, 1)
        end   = datetime(2023, 12, 31)
        days  = (end - start).days
        offsets = self.rng.integers(0, days * 24 * 60, self.n_samples)
        return pd.to_datetime([start + timedelta(minutes=int(m)) for m in offsets])

    # ── Weather ──────────────────────────────────────────
    def _generate_weather(self, months: np.ndarray) -> pd.DataFrame:
        condition = []
        temp, vis, wind, precip, humidity = [], [], [], [], []

        for m in months:
            # Season-based probabilities
            if m in [12, 1, 2]:          # Winter / fog
                cond = self.rng.choice(WEATHER_CONDITIONS, p=[0.35, 0.30, 0.05, 0.20, 0.02, 0.08])
                t  = self.rng.normal(15, 5)
                v  = self.rng.uniform(0.5, 8) if cond == "Fog" else self.rng.uniform(5, 15)
                w  = self.rng.uniform(5, 25)
                pr = 0
                h  = self.rng.uniform(70, 95)
            elif m in [6, 7, 8, 9]:      # Monsoon
                cond = self.rng.choice(WEATHER_CONDITIONS, p=[0.10, 0.20, 0.40, 0.05, 0.15, 0.10])
                t  = self.rng.normal(30, 3)
                v  = self.rng.uniform(2, 10)
                w  = self.rng.uniform(15, 50)
                pr = self.rng.exponential(10) if cond in ["Rain", "Thunderstorm"] else 0
                h  = self.rng.uniform(75, 95)
            else:                         # Other
                cond = self.rng.choice(WEATHER_CONDITIONS, p=[0.55, 0.20, 0.10, 0.03, 0.05, 0.07])
                t  = self.rng.normal(28, 6)
                v  = self.rng.uniform(8, 20)
                w  = self.rng.uniform(5, 30)
                pr = 0
                h  = self.rng.uniform(40, 70)

            condition.append(cond)
            temp.append(round(np.clip(t, 5, 48), 1))
            vis.append(round(np.clip(v, 0.1, 20), 1))
            wind.append(round(np.clip(w, 0, 60), 1))
            precip.append(round(np.clip(pr, 0, 100), 1))
            humidity.append(round(np.clip(h, 20, 100), 1))

        return pd.DataFrame({
            "weather_condition": condition,
            "temperature": temp,
            "visibility": vis,
            "wind_speed": wind,
            "precipitation": precip,
            "humidity": humidity,
        })

    # ── Delay calculation ────────────────────────────────
    def _compute_delay(self, row: pd.Series) -> float:
        base = 0.0

        # Weather impact
        if row["weather_condition"] == "Fog":       base += self.rng.uniform(20, 90)
        elif row["weather_condition"] == "Thunderstorm": base += self.rng.uniform(30, 120)
        elif row["weather_condition"] == "Rain":    base += self.rng.uniform(10, 45)
        elif row["weather_condition"] == "Haze":    base += self.rng.uniform(5, 20)

        if row["wind_speed"] > 40:    base += self.rng.uniform(15, 50)
        elif row["wind_speed"] > 25:  base += self.rng.uniform(5, 20)
        if row["visibility"] < 1:     base += self.rng.uniform(30, 80)
        elif row["visibility"] < 3:   base += self.rng.uniform(10, 40)

        # Time of day
        if row["hour"] in [7, 8, 9, 17, 18, 19, 20]:
            base += self.rng.uniform(5, 25)
        if row["day_of_week"] in [4, 5, 6]:
            base += self.rng.uniform(3, 15)

        # Season
        if row["month"] in [6, 7, 8, 9]:  base += self.rng.uniform(5, 30)
        if row["month"] in [12, 1, 2]:     base += self.rng.uniform(5, 40)

        # Random noise
        base += self.rng.exponential(3)
        base = max(0, base + self.rng.normal(0, 5))
        return round(base, 1)

    # ── Main collector ───────────────────────────────────
    def collect(self) -> pd.DataFrame:
        print(f"Generating {self.n_samples} synthetic flight records...")
        dates    = self._generate_dates()
        routes   = [ROUTE_PAIRS[i] for i in self.rng.integers(0, len(ROUTE_PAIRS), self.n_samples)]
        origins  = [r[0] for r in routes]
        dests    = [r[1] for r in routes]

        months = np.array([d.month for d in dates])
        weather_df = self._generate_weather(months)

        df = pd.DataFrame({
            "flight_number":       [f"FL{1000 + i}" for i in range(self.n_samples)],
            "airline":             self.rng.choice(AIRLINES, self.n_samples),
            "origin":              origins,
            "destination":         dests,
            "scheduled_departure": dates,
            "aircraft_type":       self.rng.choice(AIRCRAFT_TYPES, self.n_samples),
            "hour":                [d.hour for d in dates],
            "day_of_week":         [d.weekday() for d in dates],
            "month":               months.tolist(),
            "route_congestion":    self.rng.uniform(0, 1, self.n_samples).round(2),
        })

        df = pd.concat([df.reset_index(drop=True), weather_df.reset_index(drop=True)], axis=1)
        df["delay_minutes"] = df.apply(self._compute_delay, axis=1)
        df["is_delayed"]    = (df["delay_minutes"] > 15).astype(int)

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/flight_data.csv", index=False)
        print(f"  Saved → data/flight_data.csv  |  Delay rate: {df.is_delayed.mean():.1%}")
        return df


if __name__ == "__main__":
    collector = FlightDataCollector(n_samples=5000)
    df = collector.collect()
    print(df.head())
    print(df.describe())
