import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# 🎨 PAGE CONFIG + DARK UI
# =========================
st.set_page_config(page_title="Smart Crop AI", layout="wide")

st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.metric-card {
    background: #161B22;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 0 10px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# =========================
# 🌍 LOCATION
# =========================
def get_coordinates(city):
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        res = requests.get(url, timeout=5).json()
        return res["results"][0]["latitude"], res["results"][0]["longitude"]
    except:
        return 28.61, 77.23


def get_user_location():
    try:
        res = requests.get("https://ipinfo.io/json", timeout=5).json()
        lat, lon = map(float, res["loc"].split(","))
        return lat, lon, res.get("city", "Delhi")
    except:
        return 28.61, 77.23, "Delhi"

# =========================
# 🌦️ WEATHER
# =========================
def get_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=precipitation_sum"
        res = requests.get(url, timeout=5).json()

        temp = res.get("current_weather", {}).get("temperature", 25)
        rain = res.get("daily", {}).get("precipitation_sum", [0])[0]

        return temp, rain
    except:
        return 25, 0

# =========================
# 🤖 LOAD DATA + TRAIN MODEL
# =========================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "data", "crop_data.csv")

    df = pd.read_csv(file_path)

    # ✅ CLEAN COLUMNS
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # ✅ FIX FOR YOUR DATASET
    df.rename(columns={
        "Item": "Crop",
        "hg/ha_yield": "Yield"
    }, inplace=True)

    # ✅ Ensure numeric yield
    df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
    df.dropna(subset=["Yield"], inplace=True)

    # ✅ FIXED SCALING (VERY IMPORTANT)
    df["Price"] = 2000 + (df["Yield"] * 0.1)
    df["Cost"] = 50000
    df["Profit"] = (df["Yield"] * df["Price"]) - df["Cost"]

    le = LabelEncoder()
    df["Crop_enc"] = le.fit_transform(df["Crop"])

    X = df[["Crop_enc", "Yield", "Price", "Cost"]]
    y = df["Profit"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le, df

model, le, df_raw = load_model()

# =========================
# 📊 SUMMARY
# =========================
df_summary = df_raw.groupby("Crop").agg({
    "Profit": "mean",
    "Yield": ["mean", "std"]
}).reset_index()

df_summary.columns = ["Crop", "Avg_Profit", "Mean_Yield", "Std_Yield"]

# ✅ FIX NaN + division issues
df_summary["Std_Yield"] = df_summary["Std_Yield"].fillna(0)
df_summary["CV"] = (
    df_summary["Std_Yield"] / df_summary["Mean_Yield"]
).replace([np.inf, -np.inf], 0).fillna(0) * 100

def classify_risk(cv):
    if cv < 15:
        return "Low Risk"
    elif cv < 30:
        return "Medium Risk"
    return "High Risk"

df_summary["Risk"] = df_summary["CV"].apply(classify_risk)

# =========================
# 🌿 SIDEBAR
# =========================
st.sidebar.title("🌾 Smart Crop AI")

use_auto = st.sidebar.checkbox("📍 Auto Location", True)

if use_auto:
    lat, lon, city = get_user_location()
else:
    city = st.sidebar.text_input("City", "Delhi")
    lat, lon = get_coordinates(city)

crop = st.sidebar.selectbox("Crop", sorted(df_summary["Crop"]))
soil = st.sidebar.selectbox("Soil", ["Loamy", "Clay", "Sandy"])

yield_input = st.sidebar.number_input("Yield (qtl/ha)", 1, 100000, 30000)
price = st.sidebar.number_input("Price (₹/qtl)", 100, 50000, 2000)
cost = st.sidebar.number_input("Cost (₹/ha)", 10000, 200000, 50000)

# =========================
# 🌦️ WEATHER
# =========================
temp, rain = get_weather(lat, lon)

# =========================
# 🤖 ML PREDICTION
# =========================
crop_encoded = le.transform([crop])[0]

pred_profit = model.predict([[crop_encoded, yield_input, price, cost]])[0]

# Soil impact
soil_factor = {"Loamy": 1.1, "Clay": 0.95, "Sandy": 0.85}
adjusted_profit = pred_profit * soil_factor[soil]

# Weather impact
if temp > 35:
    adjusted_profit *= 0.75
elif temp < 15:
    adjusted_profit *= 0.85

if rain > 10:
    adjusted_profit *= 1.1
elif rain < 2:
    adjusted_profit *= 0.8

risk_level = df_summary[df_summary["Crop"] == crop]["Risk"].values[0]

# =========================
# 🎯 UI
# =========================
st.title("🌾 Smart Crop AI System")

st.caption(f"📍 {city} | 🌡️ {temp}°C | 🌧️ {rain} mm")

col1, col2, col3 = st.columns(3)

col1.metric("💰 Predicted Profit", f"₹{pred_profit:,.0f}")
col2.metric("🌱 Adjusted Profit", f"₹{adjusted_profit:,.0f}")
col3.metric("⚠️ Risk Level", risk_level)

# =========================
# 📌 RECOMMENDATION
# =========================
def recommend(p, risk):
    if risk == "Low Risk" and p > 50000:
        return "✅ Highly Recommended"
    elif risk == "High Risk":
        return "❌ Risky Crop"
    return "⚖️ Moderate Choice"

st.success(recommend(adjusted_profit, risk_level))

# =========================
# 🏆 TOP CROPS
# =========================
st.subheader("🏆 Top Crops")
top = df_summary.sort_values("Avg_Profit", ascending=False).head(5)
st.dataframe(top)

# =========================
# 📋 DATA
# =========================
st.subheader("📋 Dataset")
st.dataframe(df_summary)
