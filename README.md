# ✈️ AI-Powered Flight Delay Prediction System

> **Internship Project** — NTPL Digital Private Limited, Noida, UP, India
> **Intern:** Aditya Khare | UID: O24BCA160192
> **Mentor:** Vasanthi Chandran

---

## 📌 Project Overview

An end-to-end machine learning system that predicts flight delays using historical flight data, weather information, and operational metrics. The system helps passengers and airport staff manage schedules proactively.

**Key Results:**
- 🎯 **87.3% Accuracy** (XGBoost Classifier)
- 📉 **12.4 min MAE** (Delay Duration Regressor)
- 📈 **0.91 AUC-ROC Score**

---

## 🗂️ Project Structure

```
flight-delay-prediction/
├── src/
│   ├── data_collection.py      # Data generation & collection module
│   ├── preprocessing.py        # Data cleaning & preprocessing
│   ├── feature_engineering.py  # Feature creation (16+ features)
│   ├── model.py                # XGBoost training pipeline
│   ├── predict.py              # Prediction interface
│   └── dashboard.py            # Streamlit dashboard
├── api/
│   └── app.py                  # FastAPI REST endpoint
├── scripts/
│   └── train.py                # End-to-end training script
├── models/                     # Saved model files (auto-created)
├── data/                       # Dataset files (auto-created)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Aditya-gif938/flight-delay-prediction.git
cd flight-delay-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Models
```bash
python scripts/train.py
```

### 4. Launch the Dashboard
```bash
streamlit run src/dashboard.py
```

### 5. Start the REST API
```bash
uvicorn api.app:app --reload
# Docs at http://localhost:8000/docs
```

---

## 📊 Methodology (CRISP-DM)

| Phase | Description |
|-------|-------------|
| Business Understanding | Flight delay problem analysis |
| Data Understanding | Explore flight & weather datasets |
| Data Preparation | Feature engineering & preprocessing |
| Modeling | XGBoost classifier + regressor |
| Evaluation | Accuracy, F1, MAE analysis |
| Deployment | Streamlit dashboard + FastAPI |

---

## 🧠 ML Models

### Classification Model (Will the flight be delayed?)
- **Algorithm:** XGBoost Classifier
- **Threshold:** Delay > 15 minutes
- **Accuracy:** 87.3% | **F1:** 0.84 | **AUC-ROC:** 0.91

### Regression Model (How long will the delay be?)
- **Algorithm:** XGBoost Regressor
- **MAE:** 12.4 minutes | **RMSE:** 18.7 min | **R²:** 0.78

---

## 🔧 Feature Engineering (16+ Features)

| Category | Features |
|----------|----------|
| Temporal | hour, day_of_week, month, is_weekend, is_peak_hour, is_holiday |
| Weather | weather_risk, is_monsoon, is_fog_season, visibility, wind_speed |
| Route | route_congestion, airline, origin, destination, aircraft_type |

**Weather Risk Formula:**
```
Risk = 0.8×Fog + 0.3×Rain + 0.4×(Wind>20 km/h) + 0.6×(Visibility<3 km)
```

---

## 📈 Model Comparison

| Model | Accuracy | F1-Score | AUC | MAE (min) |
|-------|----------|----------|-----|-----------|
| Logistic Regression | 72.1% | 0.68 | 0.74 | 22.3 |
| Decision Tree | 78.5% | 0.75 | 0.77 | 18.9 |
| Random Forest | 84.6% | 0.81 | 0.87 | 14.1 |
| **XGBoost** ⭐ | **87.3%** | **0.84** | **0.91** | **12.4** |

---

## 🖥️ Dashboard Features

- Flight delay probability gauge (color-coded risk levels)
- Operations overview with charts
- Critical flight alerts
- Real-time prediction interface

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict delay for a flight |
| GET | `/health` | API health check |
| GET | `/docs` | Swagger UI |

**Sample Request:**
```json
POST /predict
{
  "airline": "IndiGo",
  "origin": "DEL",
  "destination": "BOM",
  "scheduled_hour": 8,
  "day_of_week": 0,
  "month": 7,
  "temperature": 35.0,
  "visibility": 8.0,
  "wind_speed": 15.0,
  "weather_condition": "Clear"
}
```

**Sample Response:**
```json
{
  "delay_probability": 0.73,
  "predicted_delay_minutes": 25.0,
  "risk_level": "High",
  "recommendation": "Consider buffer time"
}
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML | XGBoost, Scikit-learn |
| Data | Pandas, NumPy |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, Uvicorn |
| Serialization | Joblib |
| Version Control | Git / GitHub |

---

## 📡 Data Sources

- **Bureau of Transportation Statistics (BTS)** — Historical flight data
- **OpenWeatherMap API** — Real-time weather data
- **OpenFlights Database** — Airport & airline information
- **IMD Archives** — Indian Meteorological Department data

---

## 🚀 Future Scope

- **Short-Term:** Live aviation API integration, automated retraining
- **Medium-Term:** LSTM/Transformer models, mobile app
- **Long-Term:** Cloud deployment (AWS/GCP), Apache Kafka streaming

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

*Developed during internship at NTPL Digital Private Limited, Noida, UP, India*
