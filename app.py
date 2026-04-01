import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# 🎨 PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Crop AI", layout="wide")

# =========================
# 🎨 GLOBAL UI
# =========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}

/* Smooth animation */
section.main > div {
    animation: fadeIn 0.5s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    padding: 1.5rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
    transition: all 0.25s ease;
}
.card:hover { transform: translateY(-5px); }

.metric { font-size: 2rem; font-weight: bold; }

.hero {
    text-align:center;
    font-size:2.8rem;
    font-weight:800;
    background: linear-gradient(90deg,#3b82f6,#22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.weather-card {
    background: linear-gradient(90deg,#1e293b,#0f172a);
    padding: 1rem;
    border-radius: 14px;
    text-align:center;
    margin-bottom: 1rem;
}

.highlight {
    background: linear-gradient(90deg,#22c55e,#16a34a);
    padding: 1rem;
    border-radius: 12px;
    text-align:center;
    font-weight: bold;
}

.topbar {
    position: sticky;
    top: 0;
    background: rgba(2,6,23,0.9);
    padding: 10px;
    z-index: 999;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.status { color: #22c55e; font-weight: bold; }

img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 🌍 LOCATION + WEATHER
# =========================
def get_coordinates(city):
    try:
        res = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1").json()
        return res["results"][0]["latitude"], res["results"][0]["longitude"]
    except:
        return 28.61, 77.23

def get_user_location():
    try:
        res = requests.get("https://ipinfo.io/json").json()
        lat, lon = map(float, res["loc"].split(","))
        return lat, lon, res.get("city", "Delhi")
    except:
        return 28.61, 77.23, "Delhi"

def get_weather(lat, lon):
    try:
        res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=precipitation_sum").json()
        return res["current_weather"]["temperature"], res["daily"]["precipitation_sum"][0]
    except:
        return 25, 0

# =========================
# 📊 DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/crop_data.csv")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df.rename(columns={"Item": "Crop", "hg/ha_yield": "Yield"}, inplace=True)
    df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
    df.dropna(subset=["Yield"], inplace=True)
    return df

df = load_data()

df_summary = df.groupby("Crop").agg({"Yield": ["mean", "std"]}).reset_index()
df_summary.columns = ["Crop", "Mean_Yield", "Std_Yield"]
df_summary["Std_Yield"] = df_summary["Std_Yield"].fillna(0)
df_summary["CV"] = (df_summary["Std_Yield"] / df_summary["Mean_Yield"]) * 100

low = df_summary["CV"].quantile(0.33)
high = df_summary["CV"].quantile(0.66)

def risk(cv):
    if cv <= low: return "Low Risk"
    elif cv <= high: return "Medium Risk"
    return "High Risk"

df_summary["Risk"] = df_summary["CV"].apply(risk)

# =========================
# 🌿 SIDEBAR
# =========================
st.sidebar.markdown("## 🌾 Smart Crop AI")
st.sidebar.caption("AI-powered crop decision system")

page = st.sidebar.radio("Navigation", ["🏠 Home","🔮 Prediction","📊 Analytics","📄 Report"])

use_auto = st.sidebar.checkbox("📍 Auto Location", True)

if use_auto:
    lat, lon, city = get_user_location()
else:
    city = st.sidebar.text_input("City","Delhi")
    lat, lon = get_coordinates(city)

temp, rain = get_weather(lat, lon)

# =========================
# 🔝 TOP BAR
# =========================
st.markdown(f"""
<div class="topbar">
🌾 Smart Crop AI | 📍 {city} | 🌡️ {temp}°C | <span class="status">● LIVE</span>
</div>
""", unsafe_allow_html=True)

# =========================
# 🏠 HOME
# =========================
if page == "🏠 Home":

    with st.spinner("Loading Smart Crop AI..."):
        import time; time.sleep(1)

    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)

    st.markdown("<div class='hero'>🌾 Smart Crop AI</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='weather-card'>📍 {city} | 🌡️ {temp}°C | 🌧️ {rain} mm</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>📌 About the Project</h3>
    Smart Crop Profit & Risk Prediction System is an AI-driven agricultural decision system
    helping farmers choose optimal crops using weather + risk analysis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>👨‍💻 Project Details</h3>
    Developed by: Agrim Singh, Alisha Dhakal, Rajdeep Dey<br>
    Institute: SMIT<br>
    Under Guidance: Mr. Gaurav Sarma
    </div>
    """, unsafe_allow_html=True)

# =========================
# 🔮 PREDICTION
# =========================
elif page == "🔮 Prediction":

    crop = st.selectbox("Crop", sorted(df_summary["Crop"]))
    soil = st.selectbox("Soil", ["Loamy","Clay","Sandy"])

    yield_input = st.number_input("Yield",1,100000,30000)
    price = st.number_input("Price",100,50000,2000)
    cost = st.number_input("Cost",10000,200000,50000)

    avg_yield = df_summary[df_summary["Crop"]==crop]["Mean_Yield"].values[0]
    pred = ((yield_input/(avg_yield+1))*avg_yield*price)-cost

    soil_factor = {"Loamy":1.1,"Clay":0.95,"Sandy":0.85}
    adjusted = pred * soil_factor[soil]

    if temp > 35: adjusted *= 0.75
    elif temp < 15: adjusted *= 0.85
    if rain > 10: adjusted *= 1.1
    elif rain < 2: adjusted *= 0.8

    risk_level = df_summary[df_summary["Crop"]==crop]["Risk"].values[0]

    st.markdown(f"<div class='weather-card'>📍 {city} | 🌡️ {temp}°C | 🌧️ {rain} mm</div>", unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)
    col1.markdown(f"<div class='card'><h4>Profit</h4><div class='metric'>₹{pred:,.0f}</div></div>",unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><h4>Adjusted</h4><div class='metric'>₹{adjusted:,.0f}</div></div>",unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><h4>Risk</h4><div class='metric'>{risk_level}</div></div>",unsafe_allow_html=True)

    if adjusted > 0:
        st.success("✅ Recommended Crop")
    else:
        st.error("⚠️ Not Recommended")

    st.markdown(f"""
    <div class="card">
    <h3>🧠 Why this crop?</h3>
    Base Yield: {avg_yield:.2f}<br>
    Weather & soil adjustments applied<br>
    Risk: {risk_level}
    </div>
    """, unsafe_allow_html=True)

    st.session_state["report"] = (crop,pred,adjusted,risk_level,city)

# =========================
# 📊 ANALYTICS
# =========================
elif page == "📊 Analytics":

    st.subheader("📊 Analytics Dashboard")

    # Historical
    fig1 = px.bar(df_summary, x="Crop", y="Mean_Yield", color="Risk")
    fig1.update_layout(template="plotly_dark")
    st.plotly_chart(fig1)

# =========================
# 📄 REPORT
# =========================
elif page == "📄 Report":

    if "report" in st.session_state:
        crop,pred,adjusted,risk_level,city = st.session_state["report"]

        file="report.pdf"
        doc=SimpleDocTemplate(file)
        styles=getSampleStyleSheet()

        doc.build([
            Paragraph(f"Crop: {crop}",styles["Normal"]),
            Paragraph(f"Profit: ₹{pred:,.0f}",styles["Normal"]),
            Paragraph(f"Adjusted: ₹{adjusted:,.0f}",styles["Normal"]),
            Paragraph(f"Risk: {risk_level}",styles["Normal"])
        ])

        with open(file,"rb") as f:
            st.download_button("Download Report",f,"report.pdf")
