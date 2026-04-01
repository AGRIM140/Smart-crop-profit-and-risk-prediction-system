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

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Hero */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg,#3b82f6,#22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    padding: 1.5rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    transition: all 0.2s ease;
}

.card:hover {
    transform: translateY(-6px);
}

.metric {
    font-size: 2rem;
    font-weight: bold;
}

.section {
    margin-top: 2rem;
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
# 📊 DATA
# =========================
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "crop_data.csv"))

    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    df.rename(columns={
        "Item": "Crop",
        "hg/ha_yield": "Yield"
    }, inplace=True)

    df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
    df.dropna(subset=["Yield"], inplace=True)

    return df

df = load_data()

# =========================
# 📊 SUMMARY
# =========================
df_summary = df.groupby("Crop").agg({
    "Yield": ["mean", "std"]
}).reset_index()

df_summary.columns = ["Crop", "Mean_Yield", "Std_Yield"]

df_summary["Std_Yield"] = df_summary["Std_Yield"].fillna(0)
df_summary["CV"] = (df_summary["Std_Yield"] / df_summary["Mean_Yield"]).replace([np.inf,-np.inf],0)*100

# Dynamic risk
low = df_summary["CV"].quantile(0.33)
high = df_summary["CV"].quantile(0.66)

def risk(cv):
    if cv <= low: return "Low Risk"
    elif cv <= high: return "Medium Risk"
    return "High Risk"

df_summary["Risk"] = df_summary["CV"].apply(risk)

# =========================
# 🌿 NAV
# =========================
st.sidebar.title("🌾 Smart Crop AI")

page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "🔮 Prediction",
    "📊 Analytics",
    "📄 Report"
])

use_auto = st.sidebar.checkbox("📍 Auto Location", True)

if use_auto:
    lat, lon, city = get_user_location()
else:
    city = st.sidebar.text_input("City", "Delhi")
    lat, lon = get_coordinates(city)

temp, rain = get_weather(lat, lon)

# =========================
# 🏠 HOME
# =========================
if page == "🏠 Home":
    st.markdown("<div class='hero'>🌾 Smart Crop AI</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="weather-card">
    📍 <b>{city}</b> &nbsp;&nbsp; 🌡️ <b>{temp}°C</b> &nbsp;&nbsp; 🌧️ {rain} mm
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>🚀 Features</h3>
    <ul>
    <li>Real-time weather-aware predictions</li>
    <li>Dynamic risk classification</li>
    <li>Advanced analytics dashboard</li>
    <li>Decision support system</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =========================
# 🔮 PREDICTION
# =========================
elif page == "🔮 Prediction":

    crop = st.selectbox("Crop", sorted(df_summary["Crop"]))
    soil = st.selectbox("Soil", ["Loamy","Clay","Sandy"])

    yield_input = st.number_input("Yield", 1, 100000, 30000)
    price = st.number_input("Price", 100, 50000, 2000)
    cost = st.number_input("Cost", 10000, 200000, 50000)

    avg_yield = df_summary[df_summary["Crop"] == crop]["Mean_Yield"].values[0]

    pred = ((yield_input / (avg_yield + 1)) * avg_yield * price) - cost

    soil_factor = {"Loamy":1.1,"Clay":0.95,"Sandy":0.85}
    adjusted = pred * soil_factor[soil]

    if temp > 35: adjusted *= 0.75
    elif temp < 15: adjusted *= 0.85
    if rain > 10: adjusted *= 1.1
    elif rain < 2: adjusted *= 0.8

    risk_level = df_summary[df_summary["Crop"]==crop]["Risk"].values[0]

    st.markdown(f"""
    <div class="weather-card">
    📍 <b>{city}</b> &nbsp;&nbsp; 🌡️ <b>{temp}°C</b> &nbsp;&nbsp; 🌧️ {rain} mm
    </div>
    """, unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)

    col1.markdown(f"<div class='card'><h4>💰 Profit</h4><div class='metric'>₹{pred:,.0f}</div></div>",unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><h4>🌱 Adjusted</h4><div class='metric'>₹{adjusted:,.0f}</div></div>",unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><h4>⚠️ Risk</h4><div class='metric'>{risk_level}</div></div>",unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
    <h3>🧠 Why this crop?</h3>
    <ul>
    <li><b>Crop:</b> {crop}</li>
    <li><b>Risk:</b> {risk_level}</li>
    <li><b>Weather impact applied</b></li>
    <li><b>Soil adjustment applied</b></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.session_state["report"] = (crop, pred, adjusted, risk_level, city)

# =========================
# 📊 ANALYTICS
# =========================
elif page == "📊 Analytics":

    st.subheader("📊 Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(df_summary, x="Crop", y="Mean_Yield", color="Risk")
        fig1.update_layout(template="plotly_dark")
        st.plotly_chart(fig1)

    with col2:
        fig2 = px.pie(df_summary, names="Risk")
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2)

    fig3 = px.scatter(df_summary, x="Mean_Yield", y="CV", color="Risk", size="CV")
    fig3.update_layout(template="plotly_dark")
    st.plotly_chart(fig3)

# =========================
# 📄 REPORT
# =========================
elif page == "📄 Report":

    if "report" in st.session_state:
        crop, pred, adjusted, risk_level, city = st.session_state["report"]

        def generate_pdf():
            file="report.pdf"
            doc=SimpleDocTemplate(file)
            styles=getSampleStyleSheet()

            content=[
                Paragraph(f"Crop: {crop}",styles["Normal"]),
                Paragraph(f"Location: {city}",styles["Normal"]),
                Paragraph(f"Profit: ₹{pred:,.0f}",styles["Normal"]),
                Paragraph(f"Adjusted: ₹{adjusted:,.0f}",styles["Normal"]),
                Paragraph(f"Risk: {risk_level}",styles["Normal"])
            ]
            doc.build(content)
            return file

        pdf = generate_pdf()

        with open(pdf,"rb") as f:
            st.download_button("📥 Download Report", f, "crop_report.pdf")

    else:
        st.warning("⚠️ Run prediction first")
