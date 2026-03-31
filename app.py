import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# 🎨 PREMIUM UI
# =========================
st.set_page_config(page_title="Smart Crop AI", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0E1117, #111827);
    color: white;
}
.metric-card {
    background: rgba(22,27,34,0.7);
    padding: 1.2rem;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}
h1 {
    text-align:center;
    background: linear-gradient(90deg,#58A6FF,#22C55E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 🌍 LOCATION
# =========================
def get_coordinates(city):
    try:
        res = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1",
            timeout=5
        ).json()
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
        res = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=precipitation_sum",
            timeout=5
        ).json()
        temp = res.get("current_weather", {}).get("temperature", 25)
        rain = res.get("daily", {}).get("precipitation_sum", [0])[0]
        return temp, rain
    except:
        return 25, 0

# =========================
# 🤖 LOAD DATA + MODEL
# =========================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "crop_data.csv"))

    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    df.rename(columns={
        "Item": "Crop",
        "hg/ha_yield": "Yield"
    }, inplace=True)

    df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
    df.dropna(subset=["Yield"], inplace=True)

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
df_summary["Std_Yield"] = df_summary["Std_Yield"].fillna(0)
df_summary["CV"] = (df_summary["Std_Yield"] / df_summary["Mean_Yield"]).replace([np.inf,-np.inf],0)*100

def risk(cv):
    if cv < 15: return "Low Risk"
    elif cv < 30: return "Medium Risk"
    return "High Risk"

df_summary["Risk"] = df_summary["CV"].apply(risk)

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
soil = st.sidebar.selectbox("Soil", ["Loamy","Clay","Sandy"])

yield_input = st.sidebar.number_input("Yield", 1, 100000, 30000)
price = st.sidebar.number_input("Price", 100, 50000, 2000)
cost = st.sidebar.number_input("Cost", 10000, 200000, 50000)

# =========================
# 🌦️ WEATHER
# =========================
temp, rain = get_weather(lat, lon)

# =========================
# 🤖 PREDICTION
# =========================
crop_encoded = le.transform([crop])[0]
pred = model.predict([[crop_encoded, yield_input, price, cost]])[0]

soil_factor = {"Loamy":1.1,"Clay":0.95,"Sandy":0.85}
adjusted = pred * soil_factor[soil]

if temp > 35: adjusted *= 0.75
elif temp < 15: adjusted *= 0.85

if rain > 10: adjusted *= 1.1
elif rain < 2: adjusted *= 0.8

risk_level = df_summary[df_summary["Crop"]==crop]["Risk"].values[0]

# =========================
# 🎯 UI
# =========================
st.markdown("<h1>🌾 Smart Crop AI System</h1>", unsafe_allow_html=True)
st.caption(f"📍 {city} | 🌡️ {temp}°C | 🌧️ {rain} mm")

col1,col2,col3 = st.columns(3)

col1.markdown(f"<div class='metric-card'><h3>💰 Profit</h3><h2>₹{pred:,.0f}</h2></div>",unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><h3>🌱 Adjusted</h3><h2>₹{adjusted:,.0f}</h2></div>",unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><h3>⚠️ Risk</h3><h2>{risk_level}</h2></div>",unsafe_allow_html=True)

# =========================
# 🧠 WHY THIS CROP
# =========================
st.subheader("🧠 Why this crop?")
st.info(f"""
Crop: {crop}  
Risk: {risk_level}  
Weather impact applied  
Soil: {soil}  
Profit adjusted based on real conditions
""")

# =========================
# 📊 CHART
# =========================
st.subheader("📊 Profit Comparison")

fig = px.bar(df_summary.sort_values("Avg_Profit",ascending=False).head(10),
             x="Crop", y="Avg_Profit", color="Risk")

fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# =========================
# 📄 PDF
# =========================
def generate_pdf():
    file="report.pdf"
    doc=SimpleDocTemplate(file)
    styles=getSampleStyleSheet()

    content=[
        Paragraph(f"Crop: {crop}",styles["Normal"]),
        Paragraph(f"Profit: ₹{pred:,.0f}",styles["Normal"]),
        Paragraph(f"Adjusted: ₹{adjusted:,.0f}",styles["Normal"]),
        Paragraph(f"Risk: {risk_level}",styles["Normal"])
    ]
    doc.build(content)
    return file

pdf=generate_pdf()

with open(pdf,"rb") as f:
    st.download_button("📥 Download Report", f, "crop_report.pdf")

# =========================
# 🏆 DATA
# =========================
st.subheader("🏆 Top Crops")
st.dataframe(df_summary.sort_values("Avg_Profit",ascending=False).head(5))
