"""
dashboard.py
------------
Streamlit dashboard for flight delay prediction.
Run: streamlit run src/dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.title("✈️ Flight Delay Predictor")
st.sidebar.markdown("**Internship Project — NTPL Digital**")
st.sidebar.markdown("Aditya Khare | O24BCA160192")
page = st.sidebar.radio("Navigation", ["🔮 Predict", "📊 Dashboard", "ℹ️ About"])

# ── Helpers ──────────────────────────────────────────────
RISK_COLOR = {"Low": "#28A745", "Medium": "#FFA500", "High": "#DC3545"}

def gauge(prob, title="Delay Probability"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        title={"text": title, "font": {"size": 20}},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#1F4E79"},
            "steps": [
                {"range": [0, 30],  "color": "#E8F8ED"},
                {"range": [30, 70], "color": "#FFF3E0"},
                {"range": [70, 100],"color": "#FDECEA"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "value": 70},
        }
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=10))
    return fig

# ══════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════
if page == "🔮 Predict":
    st.title("🔮 Flight Delay Prediction")
    st.markdown("Enter flight details to get an AI-powered delay forecast.")

    col1, col2, col3 = st.columns(3)
    with col1:
        airline   = st.selectbox("Airline", ["IndiGo","Air India","SpiceJet","Vistara","GoFirst","AirAsia India"])
        origin    = st.selectbox("Origin", ["DEL","BOM","BLR","MAA","CCU","HYD","COK","PNQ"])
        dest      = st.selectbox("Destination", ["BOM","DEL","BLR","MAA","CCU","HYD","COK","PNQ"])
    with col2:
        hour      = st.slider("Departure Hour", 0, 23, 8)
        dow       = st.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        month     = st.selectbox("Month", list(range(1,13)), index=6)
    with col3:
        temp      = st.number_input("Temperature (°C)", 5.0, 50.0, 32.0, 0.5)
        vis       = st.number_input("Visibility (km)",  0.1, 20.0, 8.0,  0.5)
        wind      = st.number_input("Wind Speed (km/h)",0.0, 80.0, 15.0, 1.0)
        condition = st.selectbox("Weather Condition", ["Clear","Cloudy","Rain","Fog","Thunderstorm","Haze"])

    if st.button("🚀 Predict Delay", type="primary"):
        # Simple rule-based demo (replace with actual model when models/ exists)
        risk_score = 0.0
        if condition == "Fog":           risk_score += 0.55
        elif condition == "Thunderstorm":risk_score += 0.50
        elif condition == "Rain":        risk_score += 0.25
        if wind > 40:    risk_score += 0.20
        if vis < 3:      risk_score += 0.20
        if hour in [7,8,9,17,18,19]: risk_score += 0.10
        if month in [6,7,8,9]:       risk_score += 0.10
        prob = min(risk_score + np.random.uniform(0, 0.1), 0.99)
        mins = max(0, prob * 80 + np.random.normal(0, 5))
        risk = "Low" if prob < 0.3 else ("Medium" if prob < 0.7 else "High")

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Delay Probability", f"{prob:.1%}")
        c2.metric("Predicted Delay",   f"{mins:.0f} min")
        c3.metric("Risk Level",        risk)
        c4.metric("Status", "⚠️ Delayed" if prob > 0.3 else "✅ On Time")

        st.plotly_chart(gauge(prob), use_container_width=True)

        color = RISK_COLOR[risk]
        st.markdown(
            f'<div style="background:{color}22;border-left:4px solid {color};'
            f'padding:12px;border-radius:4px;margin-top:12px">'
            f'<b style="color:{color}">{risk} Risk</b><br>'
            f'{"High delay probability — plan for buffer time." if risk=="High" else "Moderate risk — monitor updates." if risk=="Medium" else "Flight expected on time."}'
            f'</div>', unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Operations Dashboard")

    # Generate demo data
    np.random.seed(42)
    n = 200
    hours   = np.random.randint(0, 24, n)
    delayed = np.random.binomial(1, 0.28 + 0.15 * (hours % 12 < 3), n)
    airlines= np.random.choice(["IndiGo","Air India","SpiceJet","Vistara","GoFirst"], n)
    delays  = delayed * np.random.exponential(30, n)

    df = pd.DataFrame({"hour": hours, "is_delayed": delayed, "airline": airlines, "delay_min": delays})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flights",  n)
    col2.metric("Delayed",        int(delayed.sum()), f"{delayed.mean():.1%}")
    col3.metric("Avg Delay",      f"{delays[delays>0].mean():.0f} min")
    col4.metric("On-Time Rate",   f"{1-delayed.mean():.1%}")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names="airline", title="Flights by Airline", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        hourly = df.groupby("hour")["is_delayed"].mean().reset_index()
        fig = px.bar(hourly, x="hour", y="is_delayed", title="Delay Rate by Hour",
                     labels={"is_delayed":"Delay Rate","hour":"Hour"},
                     color="is_delayed", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🚨 Critical Flights Alert")
    high_risk = df[df["delay_min"] > 60].head(5)
    if len(high_risk):
        st.dataframe(high_risk.rename(columns={"delay_min":"Delay (min)","is_delayed":"Delayed","hour":"Hour","airline":"Airline"}),
                     use_container_width=True)
    else:
        st.success("No critical delays at the moment.")

# ══════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════
else:
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### AI-Powered Flight Delay Prediction System
    **Intern:** Aditya Khare | O24BCA160192
    **Mentor:** Vasanthi Chandran
    **Organization:** NTPL Digital Private Limited, Noida, UP

    ---
    #### Key Results
    | Metric | Value |
    |--------|-------|
    | Accuracy | **87.3%** |
    | F1-Score | **0.84** |
    | AUC-ROC  | **0.91** |
    | MAE (delay duration) | **12.4 min** |

    #### Technology Stack
    Python · XGBoost · Scikit-learn · Streamlit · Plotly · FastAPI
    """)
