import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# 🔬 ADDITIONAL ML IMPORTS (ADDED ONLY)
# =========================
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import shap

# =========================
# 🎨 PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Crop AI", layout="wide")

# =========================
# 🎨 GLOBAL UI (UNCHANGED)
# =========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}
section.main > div {
    animation: fadeIn 0.5s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 1.5rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
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
.topbar {
    position: sticky;
    top: 0;
    background: rgba(2,6,23,0.9);
    padding: 10px;
}
.status { color: #22c55e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =========================
# 🌍 LOCATION + WEATHER (UNCHANGED)
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
# 📊 DATA (UNCHANGED)
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
# 🔬 MULTI-MODEL TRAINING (ADDED)
# =========================
@st.cache_resource
def train_models(df):
    df_model = df.copy()
    df_model["Price"] = 2000
    df_model["Cost"] = 50000
    df_model["Profit"] = (df_model["Yield"] * df_model["Price"]) - df_model["Cost"]

    X = df_model[["Yield", "Price", "Cost"]]
    y = df_model["Profit"]

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=120, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(),
        "LinearRegression": LinearRegression()
    }

    trained_models = {}
    scores = {}

    for name, model in models.items():
        model.fit(X, y)
        score = np.mean(cross_val_score(model, X, y, cv=5))
        trained_models[name] = model
        scores[name] = score

    return trained_models, scores

models, model_scores = train_models(df)

# =========================
# 🌿 SIDEBAR (UNCHANGED)
# =========================
st.sidebar.markdown("## 🌾 Smart Crop AI")
page = st.sidebar.radio("Navigation", ["🏠 Home","🔮 Prediction","📊 Analytics","📄 Report"])

lat, lon, city = get_user_location()
temp, rain = get_weather(lat, lon)

# =========================
# 🔝 TOP BAR (UNCHANGED)
# =========================
st.markdown(f"""
<div class="topbar">
🌾 Smart Crop AI | 📍 {city} | 🌡️ {temp}°C | <span class="status">● LIVE</span>
</div>
""", unsafe_allow_html=True)

# =========================
# 🔮 PREDICTION (ORIGINAL + ADDITIONS)
# =========================
if page == "🔮 Prediction":

    crop = st.selectbox("Crop", sorted(df_summary["Crop"]))
    soil = st.selectbox("Soil", ["Loamy","Clay","Sandy"])

    yield_input = st.number_input("Yield",1,100000,30000)
    price = st.number_input("Price",100,50000,2000)
    cost = st.number_input("Cost",10000,200000,50000)

    avg_yield = df_summary[df_summary["Crop"]==crop]["Mean_Yield"].values[0]

    # ORIGINAL (UNCHANGED)
    pred = ((yield_input/(avg_yield+1))*avg_yield*price)-cost

    # =========================
    # 🔬 MULTI-MODEL PREDICTION (ADDED)
    # =========================
    ml_results = {}
    for name, model in models.items():
        ml_results[name] = model.predict([[yield_input, price, cost]])[0]

    best_model_name = max(model_scores, key=model_scores.get)
    best_model_pred = ml_results[best_model_name]

    # =========================
    # 📊 DISPLAY (UNCHANGED + ADDED)
    # =========================
    col1,col2,col3 = st.columns(3)

    col1.markdown(f"<div class='card'><h4>Formula</h4><h2>₹{pred:,.0f}</h2></div>",unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><h4>Best ML</h4><h2>₹{best_model_pred:,.0f}</h2></div>",unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><h4>Best Model</h4><h2>{best_model_name}</h2></div>",unsafe_allow_html=True)

    # =========================
    # 📊 MODEL COMPARISON (ADDED)
    # =========================
    st.markdown("<div class='card'><h4>Model Comparison</h4>", unsafe_allow_html=True)
    for name, val in ml_results.items():
        st.write(f"{name}: ₹{val:,.0f} | Score: {model_scores[name]:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # 📊 CONFIDENCE (ADDED)
    # =========================
    confidence = 100 - abs(pred - best_model_pred)/(abs(pred)+1)*100

    st.markdown(f"""
    <div class="card">
    <h4>📊 Hybrid Confidence</h4>
    {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 🔍 SHAP EXPLAINABILITY (ADDED)
    # =========================
    explainer = shap.Explainer(models[best_model_name])
    shap_values = explainer([[yield_input, price, cost]])

    shap_df = pd.DataFrame({
        "Feature": ["Yield","Price","Cost"],
        "Impact": shap_values.values[0]
    })

    fig = px.bar(shap_df, x="Feature", y="Impact", color="Impact")
    fig.update_layout(template="plotly_dark")

    st.subheader("🔍 Explainability (Best Model)")
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 📈 FEATURE IMPORTANCE (ADDED)
    # =========================
    if best_model_name in ["RandomForest", "GradientBoosting"]:
        importances = models[best_model_name].feature_importances_

        fi_df = pd.DataFrame({
            "Feature": ["Yield","Price","Cost"],
            "Importance": importances
        })

        fig_fi = px.bar(fi_df, x="Feature", y="Importance", color="Importance")
        fig_fi.update_layout(template="plotly_dark")

        st.subheader("📈 Feature Importance")
        st.plotly_chart(fig_fi, use_container_width=True)
