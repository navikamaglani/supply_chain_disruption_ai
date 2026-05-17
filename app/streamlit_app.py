import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import requests
from datetime import datetime
import shap

from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# Config
# --------------------------------------------------
CENSUS_API_KEY = "adea131aecdf5942110d809fd31832543a48a5af"
CENSUS_IMPORTS_URL = "https://api.census.gov/data/timeseries/intltrade/imports/enduse"

BASE_DIR = "/Users/navikamaglani/Documents/personal/supply_chain_disruption_ai"

TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "trade_features.csv")
WEATHER_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "port_weather_alerts.csv")

st.set_page_config(
    page_title="Supply Chain Disruption Intelligence",
    layout="wide"
)

st.title("AI-Powered Supply Chain Disruption Intelligence System")
st.write(
    "Predicts disruption risk using automatically refreshed monthly Census trade data, "
    "live NOAA weather alerts, U.S. port dependency mapping, and explainable machine learning."
)

# --------------------------------------------------
# Helper for Charts
# --------------------------------------------------
def show_matplotlib_chart(fig):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name, bbox_inches="tight")
        st.image(tmpfile.name, use_column_width=True)
    plt.close(fig)

# --------------------------------------------------
# Fetch Latest Census Trade Data
# --------------------------------------------------
@st.cache_data(ttl=86400)
def fetch_latest_census_trade_data():
    all_data = []
    current_year = datetime.now().year

    for year in range(2020, current_year + 1):
        for month in range(1, 13):
            month_str = f"{year}-{month:02d}"

            params = {
                "get": "CTY_CODE,CTY_NAME,GEN_VAL_MO",
                "time": month_str,
                "key": CENSUS_API_KEY
            }

            try:
                response = requests.get(CENSUS_IMPORTS_URL, params=params, timeout=20)

                if response.status_code != 200:
                    continue

                data = response.json()

                if len(data) > 1:
                    temp_df = pd.DataFrame(data[1:], columns=data[0])
                    all_data.append(temp_df)

            except Exception:
                continue

    if not all_data:
        st.warning("Could not refresh Census API data. Loading saved local trade file.")
        df = pd.read_csv(TRADE_DATA_PATH)
        df["time"] = pd.to_datetime(df["time"])
        df["CTY_NAME"] = df["CTY_NAME"].astype(str)
        return df

    trade_df = pd.concat(all_data, ignore_index=True)

    trade_df["GEN_VAL_MO"] = pd.to_numeric(trade_df["GEN_VAL_MO"], errors="coerce")
    trade_df["time"] = pd.to_datetime(trade_df["time"])
    trade_df["CTY_NAME"] = trade_df["CTY_NAME"].astype(str)
    trade_df["CTY_CODE"] = trade_df["CTY_CODE"].astype(str)

    trade_df = trade_df.sort_values(["CTY_CODE", "time"])

    trade_df["year"] = trade_df["time"].dt.year
    trade_df["month"] = trade_df["time"].dt.month
    trade_df["quarter"] = trade_df["time"].dt.quarter

    trade_df["mom_change"] = trade_df.groupby("CTY_CODE")["GEN_VAL_MO"].pct_change()

    trade_df["rolling_3_month_avg"] = (
        trade_df.groupby("CTY_CODE")["GEN_VAL_MO"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    trade_df["rolling_6_month_avg"] = (
        trade_df.groupby("CTY_CODE")["GEN_VAL_MO"]
        .transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    )

    trade_df["historical_mean"] = (
        trade_df.groupby("CTY_CODE")["GEN_VAL_MO"]
        .transform(lambda x: x.expanding().mean())
    )

    trade_df["historical_std"] = (
        trade_df.groupby("CTY_CODE")["GEN_VAL_MO"]
        .transform(lambda x: x.expanding().std())
    )

    trade_df["deviation_from_avg"] = (
        (trade_df["GEN_VAL_MO"] - trade_df["historical_mean"]) /
        trade_df["historical_std"]
    )

    trade_df["high_disruption_risk"] = np.where(
        (trade_df["mom_change"] < -0.20) |
        (trade_df["deviation_from_avg"] < -1.5),
        1,
        0
    )

    trade_df = trade_df.replace([np.inf, -np.inf], np.nan)

    os.makedirs(os.path.dirname(TRADE_DATA_PATH), exist_ok=True)
    trade_df.to_csv(TRADE_DATA_PATH, index=False)

    return trade_df

@st.cache_data
def load_weather_data():
    weather_df = pd.read_csv(WEATHER_DATA_PATH)

    weather_df["port_name"] = weather_df["port_name"].astype(str)
    weather_df["state"] = weather_df["state"].astype(str)

    numeric_cols = [
        "lat",
        "lon",
        "active_alert_count",
        "warning_count",
        "watch_count",
        "advisory_count",
        "severe_weather_flag"
    ]

    for col in numeric_cols:
        weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")

    return weather_df

df = fetch_latest_census_trade_data()
weather_df = load_weather_data()

latest_trade_month = df["time"].max()
st.caption(f"Latest available Census trade data month: {latest_trade_month.strftime('%B %Y')}")

# --------------------------------------------------
# Country → U.S. Port Dependency Mapping
# --------------------------------------------------
country_port_mapping = {
    "CHINA": ["Los Angeles", "Long Beach"],
    "JAPAN": ["Los Angeles", "Seattle/Tacoma"],
    "GERMANY": ["New York/New Jersey"],
    "BRAZIL": ["Houston", "Savannah"],
    "MEXICO": ["Houston", "Los Angeles"],
    "INDIA": ["New York/New Jersey", "Savannah"],
    "CANADA": ["Seattle/Tacoma", "New York/New Jersey"],
    "UNITED KINGDOM": ["New York/New Jersey"],
    "SOUTH KOREA": ["Los Angeles", "Seattle/Tacoma"],
    "VIETNAM": ["Los Angeles", "Long Beach"],
    "TAIWAN": ["Los Angeles", "Long Beach"],
    "THAILAND": ["Los Angeles", "Long Beach"],
    "ITALY": ["New York/New Jersey"],
    "FRANCE": ["New York/New Jersey"],
    "NETHERLANDS": ["New York/New Jersey"],
    "COLOMBIA": ["Houston", "Savannah"],
    "CHILE": ["Houston", "Los Angeles"],
    "ECUADOR": ["Houston", "Savannah"]
}

# --------------------------------------------------
# Model
# --------------------------------------------------
features = [
    "GEN_VAL_MO",
    "rolling_3_month_avg",
    "rolling_6_month_avg",
    "month",
    "quarter"
]

model_df = df[
    features + ["high_disruption_risk", "CTY_NAME", "time"]
].copy()

model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

X = model_df[features]
y = model_df["high_disruption_risk"]

@st.cache_resource
def train_model(_X, _y):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(_X, _y)
    return model

@st.cache_resource
def create_shap_explainer(_model):
    return shap.TreeExplainer(_model)

model = train_model(X, y)
explainer = create_shap_explainer(model)

model_df["risk_probability"] = model.predict_proba(X)[:, 1]

model_df["risk_level"] = pd.cut(
    model_df["risk_probability"],
    bins=[0, 0.35, 0.65, 1],
    labels=["Low", "Medium", "High"],
    include_lowest=True
).astype(str)

# --------------------------------------------------
# Weather Risk
# --------------------------------------------------
weather_df["weather_risk_score"] = (
    weather_df["active_alert_count"] * 0.10 +
    weather_df["warning_count"] * 0.25 +
    weather_df["watch_count"] * 0.20 +
    weather_df["advisory_count"] * 0.10 +
    weather_df["severe_weather_flag"] * 1.00
)

weather_df["weather_risk_level"] = pd.cut(
    weather_df["weather_risk_score"],
    bins=[-1, 1, 3, 100],
    labels=["Low", "Medium", "High"]
).astype(str)

# --------------------------------------------------
# KPI Cards
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Average Trade Risk Score", f"{model_df['risk_probability'].mean():.2f}")
col2.metric("High Trade Risk Records", int((model_df["risk_level"] == "High").sum()))
col3.metric("Trade Records", len(model_df))
col4.metric("Countries / Regions", model_df["CTY_NAME"].nunique())

st.divider()

# --------------------------------------------------
# NOAA Weather Section
# --------------------------------------------------
st.subheader("Live NOAA Weather Disruption Signals by U.S. Port")

w1, w2, w3, w4 = st.columns(4)

w1.metric("Ports Monitored", weather_df["port_name"].nunique())
w2.metric("Total Active Alerts", int(weather_df["active_alert_count"].sum()))
w3.metric("Total Warnings", int(weather_df["warning_count"].sum()))
w4.metric("High Weather Risk Ports", int((weather_df["weather_risk_level"] == "High").sum()))

for _, row in weather_df.iterrows():
    message = (
        f"{row['port_name']} ({row['state']}) | "
        f"Weather Risk: {row['weather_risk_level']} | "
        f"Risk Score: {row['weather_risk_score']:.2f} | "
        f"Alerts: {int(row['active_alert_count'])} | "
        f"Warnings: {int(row['warning_count'])} | "
        f"Watches: {int(row['watch_count'])} | "
        f"Advisories: {int(row['advisory_count'])}"
    )

    if row["weather_risk_level"] == "High":
        st.error(message)
    elif row["weather_risk_level"] == "Medium":
        st.warning(message)
    else:
        st.success(message)

st.divider()

# --------------------------------------------------
# Port Map
# --------------------------------------------------
st.subheader("U.S. Port Weather Risk Map")

map_df = weather_df.rename(columns={"lat": "latitude", "lon": "longitude"})
st.map(map_df[["latitude", "longitude"]])

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Filters")

selected_region = st.sidebar.selectbox(
    "Select Trade Region / Country",
    sorted(model_df["CTY_NAME"].unique())
)

linked_ports = country_port_mapping.get(selected_region.upper(), [])

selected_port = st.sidebar.selectbox(
    "Select U.S. Port",
    sorted(weather_df["port_name"].unique())
)

# --------------------------------------------------
# Linked Port Dependencies
# --------------------------------------------------
st.subheader("Linked U.S. Port Dependencies")

if linked_ports:
    st.write(
        f"""
        The selected trade region (**{selected_region}**) is operationally connected
        with the following major U.S. ports. This helps show how global trade anomalies
        may propagate into U.S. port-level disruption risk.
        """
    )

    for port in linked_ports:
        port_info = weather_df[weather_df["port_name"] == port]

        if not port_info.empty:
            row = port_info.iloc[0]

            dependency_message = (
                f"{selected_region} trade flows may impact **{port}** operations. "
                f"Current Weather Risk: **{row['weather_risk_level']}** | "
                f"Active Alerts: {int(row['active_alert_count'])} | "
                f"Warnings: {int(row['warning_count'])}"
            )

            if row["weather_risk_level"] == "High":
                st.error(dependency_message)
            elif row["weather_risk_level"] == "Medium":
                st.warning(dependency_message)
            else:
                st.success(dependency_message)
else:
    st.info("No operational U.S. port dependency mapping available for this region yet.")

st.divider()

# --------------------------------------------------
# Trade Trend
# --------------------------------------------------
region_df = model_df[model_df["CTY_NAME"] == selected_region].copy()
region_df = region_df.sort_values("time")

for col in [
    "GEN_VAL_MO",
    "rolling_3_month_avg",
    "rolling_6_month_avg",
    "risk_probability"
]:
    region_df[col] = pd.to_numeric(region_df[col], errors="coerce")

st.subheader(f"Trade and ML Risk Trend: {selected_region}")

st.write("Monthly Trade Value and Rolling Averages")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(region_df["time"], region_df["GEN_VAL_MO"], label="Monthly Trade Value")
ax.plot(region_df["time"], region_df["rolling_3_month_avg"], label="3-Month Avg")
ax.plot(region_df["time"], region_df["rolling_6_month_avg"], label="6-Month Avg")
ax.set_xlabel("Time")
ax.set_ylabel("Trade Value")
ax.set_title("Monthly Trade Value and Rolling Averages")
ax.legend()
ax.grid(True)
show_matplotlib_chart(fig)

st.write("Disruption Risk Probability Over Time")

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(region_df["time"], region_df["risk_probability"], label="Risk Probability")
ax2.set_xlabel("Time")
ax2.set_ylabel("Risk Probability")
ax2.set_ylim(0, 1)
ax2.set_title("Disruption Risk Probability Over Time")
ax2.legend()
ax2.grid(True)
show_matplotlib_chart(fig2)

st.divider()

# --------------------------------------------------
# Selected Port Weather Details
# --------------------------------------------------
st.subheader(f"Selected Port Weather Intelligence: {selected_port}")

selected_port_df = weather_df[weather_df["port_name"] == selected_port].iloc[0]

port_summary = (
    f"{selected_port_df['port_name']} currently has "
    f"{int(selected_port_df['active_alert_count'])} active weather alerts, "
    f"{int(selected_port_df['warning_count'])} warnings, "
    f"{int(selected_port_df['watch_count'])} watches, and "
    f"{int(selected_port_df['advisory_count'])} advisories. "
    f"The current weather disruption risk level is "
    f"{selected_port_df['weather_risk_level']}."
)

if selected_port_df["weather_risk_level"] == "High":
    st.error(port_summary)
elif selected_port_df["weather_risk_level"] == "Medium":
    st.warning(port_summary)
else:
    st.success(port_summary)

# --------------------------------------------------
# Global Feature Importance
# --------------------------------------------------
st.divider()

st.subheader("Global ML Risk Drivers")

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.barh(importance_df["feature"], importance_df["importance"])
ax3.set_xlabel("Importance")
ax3.set_title("Random Forest Feature Importance")
show_matplotlib_chart(fig3)

# --------------------------------------------------
# Region-Specific SHAP Explanation
# --------------------------------------------------
st.divider()

st.subheader(f"Region-Specific Risk Explanation: {selected_region}")

latest_region_row = region_df.sort_values("time").tail(1)

if not latest_region_row.empty:
    latest_features = latest_region_row[features]
    shap_values = explainer.shap_values(latest_features)

    if isinstance(shap_values, list):
        region_shap_values = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        region_shap_values = shap_values[0, :, 1]
    else:
        region_shap_values = shap_values[0]

    shap_df = pd.DataFrame({
        "feature": features,
        "shap_value": region_shap_values,
        "feature_value": latest_features.iloc[0].values
    })

    shap_df["impact_direction"] = np.where(
        shap_df["shap_value"] > 0,
        "Increases Risk",
        "Reduces Risk"
    )

    shap_df = shap_df.reindex(
        shap_df["shap_value"].abs().sort_values(ascending=False).index
    )

    latest_risk = float(latest_region_row["risk_probability"].iloc[0])
    latest_date = latest_region_row["time"].iloc[0].date()

    st.metric(
        "Latest Region Risk Probability",
        f"{latest_risk:.2f}",
        f"As of {latest_date}"
    )

    for _, row in shap_df.iterrows():
        message = (
            f"**{row['feature']}** = {row['feature_value']:,.2f} | "
            f"Impact: {row['shap_value']:.4f} | "
            f"{row['impact_direction']}"
        )

        if row["shap_value"] > 0:
            st.error(message)
        else:
            st.success(message)

    top_driver = shap_df.iloc[0]

    st.info(
        f"""
        AI Explanation: For **{selected_region}**, the latest disruption risk is
        **{latest_risk:.2f}**. The strongest driver is **{top_driver['feature']}**,
        which currently **{top_driver['impact_direction'].lower()}**.
        """
    )
else:
    st.info("No SHAP explanation available for this region.")

# --------------------------------------------------
# Risk Simulator
# --------------------------------------------------
st.divider()

st.subheader("Disruption Risk Simulator")

default_gen_val = float(region_df["GEN_VAL_MO"].median())
default_rolling_3 = float(region_df["rolling_3_month_avg"].median())
default_rolling_6 = float(region_df["rolling_6_month_avg"].median())

c1, c2, c3 = st.columns(3)

gen_val = c1.number_input(
    "Current Monthly Trade Value",
    value=default_gen_val,
    key=f"gen_val_{selected_region}"
)

rolling_3 = c2.number_input(
    "Rolling 3-Month Average",
    value=default_rolling_3,
    key=f"rolling_3_{selected_region}"
)

rolling_6 = c3.number_input(
    "Rolling 6-Month Average",
    value=default_rolling_6,
    key=f"rolling_6_{selected_region}"
)

c4, c5 = st.columns(2)

month = c4.slider("Month", 1, 12, 6, key=f"month_{selected_region}")
quarter = c5.slider("Quarter", 1, 4, 2, key=f"quarter_{selected_region}")

input_df = pd.DataFrame([{
    "GEN_VAL_MO": gen_val,
    "rolling_3_month_avg": rolling_3,
    "rolling_6_month_avg": rolling_6,
    "month": month,
    "quarter": quarter
}])

risk_prob = model.predict_proba(input_df)[0, 1]

if risk_prob >= 0.65:
    risk_label = "High"
elif risk_prob >= 0.35:
    risk_label = "Medium"
else:
    risk_label = "Low"

st.metric("Predicted Trade Disruption Risk", f"{risk_prob:.2f}", risk_label)

if risk_label == "High":
    st.error(f"Predicted Risk Level: {risk_label}")
elif risk_label == "Medium":
    st.warning(f"Predicted Risk Level: {risk_label}")
else:
    st.success(f"Predicted Risk Level: {risk_label}")

st.info(
    f"""
    AI Summary: The selected trade scenario shows a **{risk_label} disruption risk**
    with a probability of **{risk_prob:.2f}**. The model evaluates current trade value,
    recent 3-month trade behavior, and 6-month trade trends.

    The selected port, **{selected_port}**, currently shows a
    **{selected_port_df['weather_risk_level']} weather disruption risk** based on
    NOAA active alerts, warnings, watches, advisories, and severe weather indicators.

    For mapped regions, the system links global trade flows to dependent U.S. ports
    to show how country-level trade anomalies may translate into port-level operational risk.
    """
)

# --------------------------------------------------
# Recent High-Risk Signals
# --------------------------------------------------
st.divider()

st.subheader("Recent High-Risk Trade Signals")

high_risk_df = model_df[model_df["risk_level"] == "High"].sort_values(
    "time",
    ascending=False
)

recent_records = high_risk_df[
    ["CTY_NAME", "time", "GEN_VAL_MO", "risk_probability", "risk_level"]
].head(20)

for _, row in recent_records.iterrows():
    st.write(
        f"**{row['CTY_NAME']}** | {row['time'].date()} | "
        f"Trade Value: {row['GEN_VAL_MO']:,.0f} | "
        f"Risk: {row['risk_probability']:.2f} | "
        f"Level: {row['risk_level']}"
    )