"""
╔══════════════════════════════════════════════════════════════════╗
║         SMART FARMER ASSISTANT — Noir / Gothic Edition           ║
║         Built for Sikkim Manipal Institute of Technology         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import json
import pickle
from datetime import datetime

# ── Try optional ML imports gracefully ──────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ── Try optional PDF report ─────────────────────────────────────────
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

# ════════════════════════════════════════════════════════════════════
# ⚙️  CONFIGURATION  (replace with real keys)
# ════════════════════════════════════════════════════════════════════
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY_HERE")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════════════
# 🎨  PAGE CONFIG + NOIR CSS
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Farmer Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

NOIR_CSS = """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Cinzel:wght@400;600&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

/* ── Root palette ── */
:root {
    --bg:           #050505;
    --surface:      #0d0d0d;
    --surface2:     #141414;
    --border:       #2a2a2a;
    --crimson:      #8b0000;
    --crimson-glow: #c0392b;
    --vamp:         #5c0a7d;
    --vamp-light:   #9b59b6;
    --gold:         #c9a84c;
    --text:         #e8e8e8;
    --muted:        #6b6b6b;
    --success:      #2ecc71;
    --warning:      #f39c12;
    --danger:       #e74c3c;
}

/* ── Global ── */
* { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: var(--text);
    font-family: 'EB Garamond', Georgia, serif;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #080808 !important;
    border-right: 1px solid var(--crimson);
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] label { color: var(--text) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label { color: var(--muted) !important; }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
}
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--crimson), var(--vamp)) !important;
    color: #fff !important;
    border: none !important;
    font-family: 'Cinzel', serif !important;
    letter-spacing: 2px !important;
    padding: 0.6rem 1.4rem !important;
    border-radius: 2px !important;
    text-transform: uppercase !important;
    font-size: 0.78rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 20px rgba(139,0,0,0.4) !important;
}
.stButton > button:hover {
    box-shadow: 0 0 35px rgba(139,0,0,0.8) !important;
    transform: translateY(-2px) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: var(--surface2) !important;
    color: var(--gold) !important;
    border: 1px solid var(--gold) !important;
    font-family: 'Cinzel', serif !important;
    letter-spacing: 1px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    padding: 0.75rem 1.5rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--crimson-glow) !important;
    border-bottom: 2px solid var(--crimson-glow) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--crimson) !important;
    padding: 1rem 1.25rem !important;
    border-radius: 2px !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; letter-spacing: 1px; font-family: 'Cinzel', serif; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Playfair Display', serif !important; }
[data-testid="stMetricDelta"] svg { display: none !important; }

/* ── Plotly charts ── */
.js-plotly-plot { border: 1px solid var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--crimson); border-radius: 2px; }

/* ── Custom component styles ── */
.noir-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--vamp);
    padding: 1.5rem 1.75rem;
    margin: 0.75rem 0;
    position: relative;
}
.noir-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 1px;
    background: linear-gradient(90deg, var(--vamp), transparent);
}
.noir-card-crimson {
    border-left-color: var(--crimson) !important;
}
.noir-card-crimson::before {
    background: linear-gradient(90deg, var(--crimson), transparent) !important;
}
.noir-card-gold {
    border-left-color: var(--gold) !important;
}
.noir-card-gold::before {
    background: linear-gradient(90deg, var(--gold), transparent) !important;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 5vw, 4rem);
    font-weight: 900;
    color: var(--text);
    letter-spacing: -1px;
    line-height: 1;
}
.hero-sub {
    font-family: 'Cinzel', serif;
    font-size: 0.7rem;
    letter-spacing: 4px;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.crimson-accent { color: var(--crimson-glow); }
.gold-accent { color: var(--gold); }
.vamp-accent { color: var(--vamp-light); }

.section-title {
    font-family: 'Cinzel', serif;
    font-size: 0.7rem;
    letter-spacing: 4px;
    color: var(--muted);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1.25rem;
}
.big-number {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--text);
}
.unit-label {
    font-family: 'Cinzel', serif;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}

.alert-optimal {
    background: rgba(46,204,113,0.08);
    border: 1px solid rgba(46,204,113,0.3);
    border-left: 3px solid var(--success);
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    font-family: 'Cinzel', serif;
    font-size: 0.75rem;
    letter-spacing: 1px;
    color: var(--success);
}
.alert-warning {
    background: rgba(243,156,18,0.08);
    border: 1px solid rgba(243,156,18,0.3);
    border-left: 3px solid var(--warning);
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    font-family: 'Cinzel', serif;
    font-size: 0.75rem;
    letter-spacing: 1px;
    color: var(--warning);
}
.alert-danger {
    background: rgba(231,76,60,0.08);
    border: 1px solid rgba(231,76,60,0.3);
    border-left: 3px solid var(--danger);
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    font-family: 'Cinzel', serif;
    font-size: 0.75rem;
    letter-spacing: 1px;
    color: var(--danger);
}
.weather-strip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-top: 2px solid var(--crimson);
    padding: 0.75rem 1.25rem;
    display: flex;
    gap: 2rem;
    align-items: center;
    font-family: 'Cinzel', serif;
    font-size: 0.72rem;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 1.5rem;
}
.weather-strip span { color: var(--text); }

.crop-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--crimson), var(--vamp));
    color: white;
    padding: 0.2rem 0.8rem;
    font-family: 'Cinzel', serif;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}
.vs-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1rem 0;
}
.vs-divider::before, .vs-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}
.vs-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: var(--crimson-glow);
    font-style: italic;
}
</style>
"""

st.markdown(NOIR_CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 🌍  GEOLOCATION & WEATHER APIs
# ════════════════════════════════════════════════════════════════════

def get_ip_location():
    """Auto-detect city from IP."""
    try:
        r = requests.get("https://ipinfo.io/json", timeout=4).json()
        lat, lon = map(float, r.get("loc", "28.61,77.23").split(","))
        return lat, lon, r.get("city", "Delhi"), r.get("region", "India")
    except Exception:
        return 28.61, 77.23, "Delhi", "India"

def geocode_city(city: str):
    """Convert city name → (lat, lon) via Open-Meteo geocoding."""
    try:
        r = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1",
            timeout=5
        ).json()
        res = r["results"][0]
        return res["latitude"], res["longitude"]
    except Exception:
        return 28.61, 77.23

def get_weather_openmeteo(lat: float, lon: float):
    """
    Fetch weather via Open-Meteo (free, no key needed).
    Falls back to OpenWeatherMap if API key is provided.
    Returns: temp(°C), humidity(%), rainfall(mm), wind(km/h), description
    """
    if OPENWEATHER_API_KEY and OPENWEATHER_API_KEY != "YOUR_OPENWEATHER_API_KEY_HERE":
        try:
            url = (
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
            )
            r = requests.get(url, timeout=5).json()
            temp     = r["main"]["temp"]
            humidity = r["main"]["humidity"]
            rainfall = r.get("rain", {}).get("1h", 0)
            wind     = r["wind"]["speed"] * 3.6
            desc     = r["weather"][0]["description"].title()
            return temp, humidity, rainfall, wind, desc
        except Exception:
            pass

    # Free fallback — Open-Meteo
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&hourly=relativehumidity_2m,precipitation"
            f"&timezone=auto"
        )
        r = requests.get(url, timeout=5).json()
        cw       = r.get("current_weather", {})
        temp     = cw.get("temperature", 25)
        wind     = cw.get("windspeed", 10)
        hourly   = r.get("hourly", {})
        humidity = hourly.get("relativehumidity_2m", [60])[0]
        rainfall = hourly.get("precipitation", [0])[0]
        wcode    = cw.get("weathercode", 0)
        desc_map = {0:"Clear Sky", 1:"Mainly Clear", 2:"Partly Cloudy",
                    3:"Overcast", 45:"Foggy", 61:"Light Rain", 63:"Moderate Rain",
                    65:"Heavy Rain", 80:"Showers", 95:"Thunderstorm"}
        desc = desc_map.get(wcode, "Variable")
        return float(temp), float(humidity), float(rainfall), float(wind), desc
    except Exception:
        return 25.0, 60.0, 0.0, 10.0, "Unknown"


# ════════════════════════════════════════════════════════════════════
# 📊  DATA LOADING & MODEL
# ════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_crop_data():
    """Load crop_data.csv — handle FAO-style or NPK-style schemas."""
    candidates = [
        os.path.join(BASE_DIR, "data", "crop_data.csv"),
        os.path.join(BASE_DIR, "crop_data.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            return df

    # ── Synthetic fallback (works offline / for demo) ──
    np.random.seed(42)
    crops_data = {
        "Rice":    dict(N=80,  P=40,  K=40,  ph=6.0, temp_lo=20, temp_hi=35, hum_lo=60, hum_hi=85, rain_lo=150, rain_hi=300, yield_kg=4500, cost=35000),
        "Wheat":   dict(N=60,  P=40,  K=40,  ph=6.5, temp_lo=12, temp_hi=25, hum_lo=40, hum_hi=65, rain_lo=50,  rain_hi=100, yield_kg=3800, cost=28000),
        "Maize":   dict(N=80,  P=40,  K=40,  ph=6.0, temp_lo=18, temp_hi=32, hum_lo=50, hum_hi=75, rain_lo=50,  rain_hi=120, yield_kg=5000, cost=30000),
        "Cotton":  dict(N=100, P=50,  K=50,  ph=6.5, temp_lo=25, temp_hi=40, hum_lo=35, hum_hi=65, rain_lo=50,  rain_hi=100, yield_kg=2200, cost=45000),
        "Sugarcane":dict(N=120,P=60,  K=60,  ph=6.5, temp_lo=20, temp_hi=38, hum_lo=60, hum_hi=85, rain_lo=150, rain_hi=250, yield_kg=70000,cost=55000),
        "Tomato":  dict(N=60,  P=30,  K=30,  ph=6.0, temp_lo=18, temp_hi=30, hum_lo=50, hum_hi=75, rain_lo=40,  rain_hi=80,  yield_kg=25000,cost=40000),
        "Potato":  dict(N=80,  P=50,  K=50,  ph=5.5, temp_lo=15, temp_hi=25, hum_lo=55, hum_hi=75, rain_lo=50,  rain_hi=100, yield_kg=20000,cost=38000),
        "Onion":   dict(N=50,  P=25,  K=25,  ph=6.0, temp_lo=13, temp_hi=28, hum_lo=40, hum_hi=70, rain_lo=30,  rain_hi=80,  yield_kg=15000,cost=32000),
        "Soybean": dict(N=40,  P=60,  K=30,  ph=6.0, temp_lo=20, temp_hi=30, hum_lo=55, hum_hi=75, rain_lo=60,  rain_hi=120, yield_kg=2800, cost=25000),
        "Chickpea":dict(N=30,  P=60,  K=40,  ph=6.0, temp_lo=15, temp_hi=28, hum_lo=35, hum_hi=65, rain_lo=30,  rain_hi=70,  yield_kg=1800, cost=22000),
        "Groundnut":dict(N=25, P=50,  K=50,  ph=6.0, temp_lo=22, temp_hi=35, hum_lo=45, hum_hi=70, rain_lo=50,  rain_hi=120, yield_kg=2500, cost=30000),
        "Mustard": dict(N=60,  P=30,  K=30,  ph=6.5, temp_lo=10, temp_hi=25, hum_lo=30, hum_hi=60, rain_lo=25,  rain_hi=60,  yield_kg=1600, cost=20000),
        "Jute":    dict(N=60,  P=30,  K=30,  ph=7.0, temp_lo=25, temp_hi=38, hum_lo=70, hum_hi=95, rain_lo=150, rain_hi=280, yield_kg=2800, cost=25000),
        "Banana":  dict(N=100, P=50,  K=100, ph=6.0, temp_lo=20, temp_hi=38, hum_lo=65, hum_hi=90, rain_lo=100, rain_hi=200, yield_kg=30000,cost=50000),
        "Mango":   dict(N=60,  P=40,  K=60,  ph=5.5, temp_lo=24, temp_hi=42, hum_lo=40, hum_hi=75, rain_lo=50,  rain_hi=130, yield_kg=8000, cost=35000),
    }
    rows = []
    for crop, vals in crops_data.items():
        for _ in range(60):
            rows.append({
                "label":    crop,
                "N":        vals["N"] + np.random.randint(-10,10),
                "P":        vals["P"] + np.random.randint(-8,8),
                "K":        vals["K"] + np.random.randint(-8,8),
                "temperature": np.random.uniform(vals["temp_lo"], vals["temp_hi"]),
                "humidity":    np.random.uniform(vals["hum_lo"],  vals["hum_hi"]),
                "rainfall":    np.random.uniform(vals["rain_lo"], vals["rain_hi"]),
                "ph":       vals["ph"] + np.random.uniform(-0.3, 0.3),
                "yield_kg_ha":     vals["yield_kg"] * np.random.uniform(0.85,1.15),
                "cost_per_ha":     vals["cost"] * np.random.uniform(0.9, 1.1),
                "price_per_kg":    np.random.uniform(10, 80),
            })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def build_model(df: pd.DataFrame):
    """Train a Random Forest crop classifier + yield/cost stats per crop."""
    feature_cols = [c for c in ["N","P","K","temperature","humidity","rainfall","ph"] if c in df.columns]
    if not feature_cols or "label" not in df.columns or not SKLEARN_OK:
        return None, None, None, {}

    le = LabelEncoder()
    y  = le.fit_transform(df["label"])
    X  = df[feature_cols].fillna(df[feature_cols].mean())

    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    # Per-crop stats
    stats = {}
    for crop in df["label"].unique():
        sub = df[df["label"] == crop]
        stats[crop] = {
            "yield_kg_ha":    sub["yield_kg_ha"].mean()    if "yield_kg_ha"    in sub else 3000,
            "cost_per_ha":    sub["cost_per_ha"].mean()    if "cost_per_ha"    in sub else 30000,
            "price_per_kg":   sub["price_per_kg"].mean()   if "price_per_kg"   in sub else 25,
            "yield_std":      sub["yield_kg_ha"].std()     if "yield_kg_ha"    in sub else 300,
        }

    return clf, le, feature_cols, stats


# ════════════════════════════════════════════════════════════════════
# 📐  FINANCIAL ADVISORY ENGINE
# ════════════════════════════════════════════════════════════════════

def financial_advisory(crop: str, land_acres: float, stats: dict,
                        temp: float, humidity: float, rainfall: float):
    """
    Returns a dict of financial metrics:
      - total_yield_kg, revenue, input_cost, profit
      - buy_price_seeds, target_sell_price
      - weather_factor, weather_alerts
    """
    if crop not in stats:
        return {}

    s = stats[crop]
    ha       = land_acres * 0.4047          # acres → hectares
    yield_ha = s["yield_kg_ha"]
    cost_ha  = s["cost_per_ha"]
    price_kg = s["price_per_kg"]

    # ── Weather factors ──────────────────────────────────────────────
    wf = 1.0
    alerts = []
    if temp > 38:
        wf *= 0.72;  alerts.append(("danger",  "🌡 Extreme Heat — Yield risk HIGH. Consider irrigation."))
    elif temp > 34:
        wf *= 0.88;  alerts.append(("warning", "🌡 Elevated Temperature — Monitor moisture levels."))
    elif temp < 12:
        wf *= 0.80;  alerts.append(("warning", "❄ Low Temperature — Frost risk. Cover young plants."))
    else:
        alerts.append(("optimal", "🌡 Temperature Optimal for Growth."))

    if humidity > 85:
        wf *= 0.90;  alerts.append(("warning", "💧 High Humidity — Fungal disease risk elevated."))
    elif humidity < 30:
        wf *= 0.85;  alerts.append(("warning", "🏜 Low Humidity — Moisture stress likely."))
    else:
        alerts.append(("optimal", "💧 Humidity Favorable."))

    if rainfall > 15:
        wf *= 0.85;  alerts.append(("warning", "🌧 Heavy Rainfall — Waterlogging & pest risk."))
    elif rainfall > 5:
        wf *= 1.05;  alerts.append(("optimal", "🌧 Rainfall Optimal — Weather optimal for sowing."))
    elif rainfall < 1:
        wf *= 0.88;  alerts.append(("warning", "☀ Dry Conditions — Supplement with irrigation."))
    else:
        alerts.append(("optimal", "🌧 Rainfall Adequate."))

    # ── Calculations ────────────────────────────────────────────────
    adj_yield_ha    = yield_ha * wf
    total_yield_kg  = adj_yield_ha * ha
    total_input_cost= cost_ha * ha

    # Seeds/fertilizer = ~35% of total input cost
    seed_fert_budget = total_input_cost * 0.35

    # Revenue at market price
    revenue = total_yield_kg * price_kg

    # Profit
    profit  = revenue - total_input_cost

    # Target sell price ensures 20% profit margin over input cost
    target_sell = (total_input_cost * 1.20) / max(total_yield_kg, 1)

    return {
        "ha":               round(ha, 2),
        "adj_yield_ha":     round(adj_yield_ha, 0),
        "total_yield_kg":   round(total_yield_kg, 0),
        "total_input_cost": round(total_input_cost, 0),
        "seed_fert_budget": round(seed_fert_budget, 0),
        "revenue":          round(revenue, 0),
        "profit":           round(profit, 0),
        "market_price_kg":  round(price_kg, 2),
        "target_sell_price":round(target_sell, 2),
        "weather_factor":   round(wf, 3),
        "alerts":           alerts,
    }


def predict_crop(clf, le, feature_cols, N, P, K, temp, hum, rain, ph, top_n=3):
    """Return top N crop predictions with probabilities."""
    if clf is None:
        return []
    X = np.array([[N, P, K, temp, hum, rain, ph]][:, :len(feature_cols)])
    # Safely build feature vector
    feat_map = {"N":N,"P":P,"K":K,"temperature":temp,"humidity":hum,"rainfall":rain,"ph":ph}
    X = np.array([[feat_map[f] for f in feature_cols]])
    proba = clf.predict_proba(X)[0]
    top_idx = np.argsort(proba)[::-1][:top_n]
    return [(le.classes_[i], round(proba[i]*100, 1)) for i in top_idx]


# ════════════════════════════════════════════════════════════════════
# 🖌  UI HELPERS
# ════════════════════════════════════════════════════════════════════

def noir_header(title: str, sub: str = ""):
    st.markdown(f"""
    <div class="hero-title">{title}</div>
    {"<div class='hero-sub'>"+sub+"</div>" if sub else ""}
    """, unsafe_allow_html=True)

def section_title(txt: str):
    st.markdown(f"<div class='section-title'>{txt}</div>", unsafe_allow_html=True)

def noir_card(content_html: str, accent="vamp"):
    cls = "noir-card" + (" noir-card-crimson" if accent=="crimson" else " noir-card-gold" if accent=="gold" else "")
    st.markdown(f"<div class='{cls}'>{content_html}</div>", unsafe_allow_html=True)

def weather_strip(city, temp, hum, rain, wind, desc):
    ts = datetime.now().strftime("%d %b %Y · %H:%M")
    st.markdown(f"""
    <div class="weather-strip">
        📍 <span>{city}</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        🌡 <span>{temp}°C</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        💧 <span>{hum}% RH</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        🌧 <span>{rain} mm</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        💨 <span>{wind:.1f} km/h</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        ☁ <span>{desc}</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        🕐 <span style="color:var(--muted)">{ts}</span>
    </div>
    """, unsafe_allow_html=True)

def smart_alert(level: str, msg: str):
    cls = {"optimal":"alert-optimal","warning":"alert-warning","danger":"alert-danger"}.get(level,"alert-warning")
    st.markdown(f"<div class='{cls}'>{msg}</div>", unsafe_allow_html=True)

def big_number_card(label, value, unit="", accent="crimson"):
    colors_map = {"crimson":"var(--crimson-glow)","gold":"var(--gold)","vamp":"var(--vamp-light)"}
    c = colors_map.get(accent, "var(--text)")
    st.markdown(f"""
    <div class="noir-card noir-card-{accent}">
        <div class="unit-label">{label}</div>
        <div class="big-number" style="color:{c}">{value}</div>
        <div class="unit-label">{unit}</div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 🏗  SESSION STATE INIT
# ════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "weather_fetched": False,
        "lat": 28.61, "lon": 77.23,
        "city": "Delhi", "region": "India",
        "temp": 25.0, "humidity": 60.0,
        "rainfall": 0.0, "wind": 10.0,
        "weather_desc": "Clear",
        "prediction_done": False,
        "top_crops": [],
        "fin_primary": {},
        "fin_secondary": {},
        "primary_crop": "",
        "secondary_crop": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ════════════════════════════════════════════════════════════════════
# 📂  LOAD DATA + MODEL
# ════════════════════════════════════════════════════════════════════

with st.spinner("Initialising intelligence systems..."):
    df      = load_crop_data()
    clf, le, feature_cols, stats = build_model(df)


# ════════════════════════════════════════════════════════════════════
# 🗂  SIDEBAR
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:900;color:#e8e8e8;line-height:1">
        🌾 Smart Farmer
    </div>
    <div style="font-family:'Cinzel',serif;font-size:0.6rem;letter-spacing:3px;color:#6b6b6b;margin-bottom:1.5rem">
        ASSISTANT · NOIR EDITION
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Location ────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📍 Location</div>", unsafe_allow_html=True)
    auto_loc = st.checkbox("Auto-detect location", value=True)

    if auto_loc:
        if not st.session_state.weather_fetched:
            lat, lon, city, region = get_ip_location()
            st.session_state.update({"lat":lat,"lon":lon,"city":city,"region":region})
    else:
        manual_city = st.text_input("City", value=st.session_state.city)
        if manual_city != st.session_state.city:
            lat2, lon2 = geocode_city(manual_city)
            st.session_state.update({"lat":lat2,"lon":lon2,"city":manual_city,"region":"India"})

    if st.button("🌦 Refresh Weather"):
        with st.spinner("Fetching live weather..."):
            t, h, r, w, d = get_weather_openmeteo(st.session_state.lat, st.session_state.lon)
            st.session_state.update({
                "temp":t,"humidity":h,"rainfall":r,"wind":w,
                "weather_desc":d,"weather_fetched":True
            })

    # Fetch on first load if not done
    if not st.session_state.weather_fetched:
        t, h, r, w, d = get_weather_openmeteo(st.session_state.lat, st.session_state.lon)
        st.session_state.update({
            "temp":t,"humidity":h,"rainfall":r,"wind":w,
            "weather_desc":d,"weather_fetched":True
        })

    st.markdown("---")

    # ── Soil inputs ─────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🧪 Soil Parameters</div>", unsafe_allow_html=True)

    col_n, col_p = st.columns(2)
    N_val = col_n.number_input("N", 0, 200, 80, help="Nitrogen (kg/ha)")
    P_val = col_p.number_input("P", 0, 150, 40, help="Phosphorous (kg/ha)")

    col_k, col_ph = st.columns(2)
    K_val  = col_k.number_input("K",  0, 200, 40, help="Potassium (kg/ha)")
    ph_val = col_ph.number_input("pH", 3.5, 9.5, 6.5, step=0.1, help="Soil pH")

    st.markdown("---")

    # ── Land size ────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🌿 Farm Details</div>", unsafe_allow_html=True)
    land_acres = st.number_input("Land Size (Acres)", 0.5, 1000.0, 5.0, step=0.5)

    st.markdown("---")
    if st.button("🔮  Analyse & Predict", use_container_width=True):
        top_crops = predict_crop(
            clf, le, feature_cols,
            N_val, P_val, K_val,
            st.session_state.temp,
            st.session_state.humidity,
            st.session_state.rainfall,
            ph_val, top_n=3
        )
        if not top_crops and stats:
            # Fallback: pick by yield
            ranked = sorted(stats.items(), key=lambda x: x[1]["yield_kg_ha"], reverse=True)
            top_crops = [(c, round(100/len(ranked),1)) for c,_ in ranked[:3]]

        st.session_state.top_crops       = top_crops
        st.session_state.primary_crop    = top_crops[0][0] if top_crops else ""
        st.session_state.secondary_crop  = top_crops[1][0] if len(top_crops)>1 else ""
        st.session_state.prediction_done = True

        if st.session_state.primary_crop:
            st.session_state.fin_primary = financial_advisory(
                st.session_state.primary_crop, land_acres, stats,
                st.session_state.temp, st.session_state.humidity, st.session_state.rainfall
            )
        if st.session_state.secondary_crop:
            st.session_state.fin_secondary = financial_advisory(
                st.session_state.secondary_crop, land_acres, stats,
                st.session_state.temp, st.session_state.humidity, st.session_state.rainfall
            )


# ════════════════════════════════════════════════════════════════════
# 🖼  MAIN CONTENT AREA
# ════════════════════════════════════════════════════════════════════

# ── Header ──────────────────────────────────────────────────────────
noir_header(
    '<span class="crimson-accent">Smart</span> Farmer Assistant',
    "AI · AGRONOMY · FINANCIAL INTELLIGENCE"
)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── Live weather strip ───────────────────────────────────────────────
weather_strip(
    st.session_state.city,
    st.session_state.temp,
    st.session_state.humidity,
    st.session_state.rainfall,
    st.session_state.wind,
    st.session_state.weather_desc,
)

# ════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  PREDICTION",
    "📊  ANALYTICS",
    "⚠️  RISK ORACLE",
    "📄  REPORT",
])


# ── TAB 1 — PREDICTION ──────────────────────────────────────────────
with tab1:
    if not st.session_state.prediction_done:
        noir_card("""
        <div style="text-align:center;padding:2rem 0">
            <div style="font-family:'Playfair Display',serif;font-size:2rem;color:#2a2a2a;margin-bottom:0.5rem">
                ◆ ◆ ◆
            </div>
            <div style="font-family:'Cinzel',serif;letter-spacing:3px;font-size:0.8rem;color:#4a4a4a">
                CONFIGURE PARAMETERS IN THE SIDEBAR<br>
                THEN PRESS  <span style="color:#8b0000">ANALYSE & PREDICT</span>
            </div>
        </div>
        """)
    else:
        top_crops   = st.session_state.top_crops
        primary     = st.session_state.primary_crop
        secondary   = st.session_state.secondary_crop
        fp          = st.session_state.fin_primary
        fs          = st.session_state.fin_secondary

        # ── AI Recommendation banner ─────────────────────────────────
        section_title("AI CROP RECOMMENDATION")
        if top_crops:
            badges = " &nbsp; ".join(
                [f"<span class='crop-badge'>#{i+1} {c} — {p}%</span>"
                 for i,(c,p) in enumerate(top_crops)]
            )
            noir_card(f"""
            <div style='font-family:"Cinzel",serif;font-size:0.65rem;letter-spacing:2px;color:var(--muted);margin-bottom:0.5rem'>
                TOP AI PREDICTIONS FOR YOUR SOIL & CLIMATE
            </div>
            {badges}
            """, accent="crimson")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── Dual crop comparison ──────────────────────────────────────
        if primary and secondary and fp and fs:
            section_title("HEAD-TO-HEAD COMPARISON")

            col_a, col_vs, col_b = st.columns([5, 1, 5])

            def render_crop_panel(col, crop, fin, is_primary=True):
                accent = "crimson" if is_primary else "vamp"
                with col:
                    tag = "◆ RECOMMENDED" if is_primary else "◈ ALTERNATIVE"
                    st.markdown(f"""
                    <div class='noir-card noir-card-{accent}'>
                        <div class='unit-label' style='color:{"var(--crimson-glow)" if is_primary else "var(--vamp-light)"}'>{tag}</div>
                        <div style='font-family:"Playfair Display",serif;font-size:1.8rem;font-weight:700;margin:0.25rem 0'>{crop}</div>
                        <div class='unit-label'>weather factor · {fin.get("weather_factor",1):.2f}×</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.metric("🌾 Total Yield", f"{fin.get('total_yield_kg',0):,.0f} kg",
                              f"{fin.get('adj_yield_ha',0):,.0f} kg/ha")
                    st.metric("💰 Revenue",     f"₹{fin.get('revenue',0):,.0f}")
                    st.metric("🏭 Input Cost",  f"₹{fin.get('total_input_cost',0):,.0f}")

                    profit = fin.get("profit", 0)
                    delta_str = "Profitable ✓" if profit>0 else "Loss ✗"
                    st.metric("📈 Net Profit",  f"₹{profit:,.0f}", delta_str)

                    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

                    # Financial cards
                    buy_b  = fin.get("seed_fert_budget", 0)
                    sell_p = fin.get("target_sell_price", 0)
                    mkt_p  = fin.get("market_price_kg", 0)
                    margin = ((sell_p - mkt_p)/max(mkt_p,1))*100

                    st.markdown(f"""
                    <div class='noir-card noir-card-gold'>
                        <div class='unit-label'>💵 Ideal Buy Budget (Seeds/Fertilizer)</div>
                        <div class='big-number' style='color:var(--gold)'>₹{buy_b:,.0f}</div>
                        <div class='unit-label'>35% of total input cost · {fin.get("ha",1):.1f} ha</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class='noir-card noir-card-crimson'>
                        <div class='unit-label'>🎯 Target Sell Price (20% Margin)</div>
                        <div class='big-number' style='color:var(--crimson-glow)'>₹{sell_p:.2f}/kg</div>
                        <div class='unit-label'>Mkt avg ₹{mkt_p:.2f}/kg · {"↑ above" if sell_p>=mkt_p else "↓ below"} market</div>
                    </div>
                    """, unsafe_allow_html=True)

            render_crop_panel(col_a, primary,   fp, is_primary=True)
            with col_vs:
                st.markdown("<div style='height:5rem'></div>", unsafe_allow_html=True)
                st.markdown("<div class='vs-divider'><div class='vs-text'>vs</div></div>", unsafe_allow_html=True)
            render_crop_panel(col_b, secondary, fs, is_primary=False)

        elif primary and fp:
            # Single crop view
            section_title("FINANCIAL INTELLIGENCE")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🌾 Yield",      f"{fp.get('total_yield_kg',0):,.0f} kg")
            c2.metric("💰 Revenue",    f"₹{fp.get('revenue',0):,.0f}")
            c3.metric("🏭 Input Cost", f"₹{fp.get('total_input_cost',0):,.0f}")
            c4.metric("📈 Net Profit", f"₹{fp.get('profit',0):,.0f}")

            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                big_number_card("💵 Ideal Buy Budget", f"₹{fp.get('seed_fert_budget',0):,.0f}", "Seeds & Fertilizers", "gold")
            with cb:
                big_number_card("🎯 Target Sell Price", f"₹{fp.get('target_sell_price',0):.2f}", "per kg · 20% margin guaranteed", "crimson")

        # ── Smart Weather Alerts ──────────────────────────────────────
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        section_title("⚡ SMART ADVISORIES")

        alerts = fp.get("alerts", []) if fp else []
        if alerts:
            for level, msg in alerts:
                smart_alert(level, msg)
        else:
            smart_alert("optimal", "✅ All conditions nominal. Proceed with confidence.")


# ── TAB 2 — ANALYTICS ───────────────────────────────────────────────
with tab2:
    import plotly.express as px
    import plotly.graph_objects as go

    section_title("CROP INTELLIGENCE ANALYTICS")

    DARK_TPL = dict(
        plot_bgcolor="#0d0d0d",
        paper_bgcolor="#0d0d0d",
        font_color="#e8e8e8",
        font_family="EB Garamond",
    )

    if stats:
        df_stats = pd.DataFrame([
            {"Crop": c, **v} for c, v in stats.items()
        ])

        # ── Chart 1 & 2 ──────────────────────────────────────────────
        c1, c2 = st.columns(2)

        with c1:
            fig = px.bar(
                df_stats.sort_values("yield_kg_ha", ascending=True),
                x="yield_kg_ha", y="Crop", orientation="h",
                title="Yield per Hectare (kg)",
                color="yield_kg_ha",
                color_continuous_scale=["#2a0000","#8b0000","#c0392b","#e74c3c"],
            )
            fig.update_layout(**DARK_TPL, title_font_size=13,
                              coloraxis_showscale=False,
                              margin=dict(l=10,r=10,t=40,b=10))
            fig.update_traces(marker_line_color="#0d0d0d", marker_line_width=0.5)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(
                df_stats.sort_values("cost_per_ha", ascending=True),
                x="cost_per_ha", y="Crop", orientation="h",
                title="Input Cost per Hectare (₹)",
                color="cost_per_ha",
                color_continuous_scale=["#0a001a","#5c0a7d","#9b59b6","#d7bde2"],
            )
            fig.update_layout(**DARK_TPL, title_font_size=13,
                              coloraxis_showscale=False,
                              margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        # ── Chart 3 — Profit vs Cost scatter ─────────────────────────
        df_stats["est_profit"] = (
            df_stats["yield_kg_ha"] * df_stats["price_per_kg"] - df_stats["cost_per_ha"]
        )
        fig = px.scatter(
            df_stats, x="cost_per_ha", y="est_profit",
            size="yield_kg_ha", color="est_profit", text="Crop",
            title="Profitability Matrix — Input Cost vs Estimated Profit",
            color_continuous_scale=["#8b0000","#f39c12","#2ecc71"],
        )
        fig.update_traces(textposition="top center", textfont_size=10,
                          marker_line_color="#0d0d0d", marker_line_width=0.5)
        fig.update_layout(**DARK_TPL, title_font_size=13,
                          coloraxis_showscale=True,
                          margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        # ── Feature importance (if model available) ───────────────────
        if clf is not None and feature_cols:
            imp = clf.feature_importances_
            fig = go.Figure(go.Bar(
                x=imp, y=feature_cols, orientation="h",
                marker_color=["#8b0000","#c0392b","#e74c3c","#5c0a7d","#9b59b6","#d7bde2","#c9a84c"][:len(feature_cols)],
                text=[f"{v:.3f}" for v in imp],
                textposition="outside",
            ))
            fig.update_layout(**DARK_TPL,
                              title="🧠 Feature Importance — Crop Prediction Model",
                              title_font_size=13, margin=dict(l=10,r=40,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)
    else:
        noir_card("<i>Analytics unavailable — load crop_data.csv to unlock.</i>")


# ── TAB 3 — RISK ORACLE ─────────────────────────────────────────────
with tab3:
    import plotly.express as px

    section_title("RISK ORACLE")

    if stats:
        df_risk = pd.DataFrame([
            {
                "Crop": c,
                "Yield Std Dev": v.get("yield_std", 0),
                "Yield (kg/ha)": v.get("yield_kg_ha", 0),
                "Cost (₹/ha)": v.get("cost_per_ha", 0),
                "CV (%)": round(v.get("yield_std",0) / max(v.get("yield_kg_ha",1),1) * 100, 1),
            }
            for c, v in stats.items()
        ])

        # Risk tier
        q33 = df_risk["CV (%)"].quantile(0.33)
        q66 = df_risk["CV (%)"].quantile(0.66)
        def risk_tier(cv):
            if cv <= q33:   return "🟢 Low Risk"
            elif cv <= q66: return "🟡 Medium Risk"
            return "🔴 High Risk"
        df_risk["Risk Tier"] = df_risk["CV (%)"].apply(risk_tier)

        DARK_TPL2 = dict(plot_bgcolor="#0d0d0d", paper_bgcolor="#0d0d0d",
                         font_color="#e8e8e8", font_family="EB Garamond")

        # Pie
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df_risk, names="Risk Tier",
                         color="Risk Tier",
                         color_discrete_map={
                             "🟢 Low Risk":"#2ecc71",
                             "🟡 Medium Risk":"#f39c12",
                             "🔴 High Risk":"#e74c3c"
                         },
                         title="Risk Distribution", hole=0.45)
            fig.update_layout(**DARK_TPL2, title_font_size=13,
                              margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(df_risk.sort_values("CV (%)", ascending=False),
                         x="Crop", y="CV (%)", color="Risk Tier",
                         color_discrete_map={
                             "🟢 Low Risk":"#2ecc71",
                             "🟡 Medium Risk":"#f39c12",
                             "🔴 High Risk":"#e74c3c"
                         },
                         title="Coefficient of Variation (%) by Crop")
            fig.update_layout(**DARK_TPL2, title_font_size=13,
                              margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Sortable risk table
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        section_title("FULL RISK TABLE")
        show_df = df_risk[["Crop","Yield (kg/ha)","Cost (₹/ha)","CV (%)","Risk Tier"]].sort_values("CV (%)")
        st.dataframe(
            show_df.style.background_gradient(subset=["CV (%)"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True
        )
    else:
        noir_card("<i>Risk analytics unavailable.</i>")


# ── TAB 4 — REPORT ──────────────────────────────────────────────────
with tab4:
    section_title("FIELD INTELLIGENCE REPORT")

    if not st.session_state.prediction_done:
        noir_card("<i style='color:#4a4a4a'>Run a prediction first to generate your report.</i>")
    else:
        fp       = st.session_state.fin_primary
        primary  = st.session_state.primary_crop
        top_c    = st.session_state.top_crops

        noir_card(f"""
        <div class='unit-label'>REPORT DATE</div>
        <div style='font-family:"Playfair Display",serif;font-size:1.1rem;margin-bottom:1rem'>
            {datetime.now().strftime("%d %B %Y, %H:%M")}
        </div>
        <div class='unit-label'>LOCATION</div>
        <div style='margin-bottom:1rem'>{st.session_state.city}, {st.session_state.region}</div>
        <div class='unit-label'>LIVE CONDITIONS</div>
        <div>🌡 {st.session_state.temp}°C &nbsp; 💧 {st.session_state.humidity}% RH &nbsp; 🌧 {st.session_state.rainfall} mm rainfall</div>
        """, accent="gold")

        if fp:
            c1, c2, c3 = st.columns(3)
            c1.metric("Primary Crop",   primary)
            c2.metric("Land (Acres)",   f"{fp.get('ha',0)/0.4047:.1f}")
            c3.metric("Area (Hectares)",f"{fp.get('ha',0):.2f}")

            c4, c5, c6 = st.columns(3)
            c4.metric("Total Yield",   f"{fp.get('total_yield_kg',0):,.0f} kg")
            c5.metric("Revenue",       f"₹{fp.get('revenue',0):,.0f}")
            c6.metric("Net Profit",    f"₹{fp.get('profit',0):,.0f}")

            c7, c8 = st.columns(2)
            c7.metric("Buy Budget",        f"₹{fp.get('seed_fert_budget',0):,.0f}")
            c8.metric("Target Sell Price", f"₹{fp.get('target_sell_price',0):.2f}/kg")

        # ── PDF Download ─────────────────────────────────────────────
        if REPORTLAB_OK and fp:
            def build_pdf():
                from io import BytesIO
                buf = BytesIO()
                doc = SimpleDocTemplate(buf, leftMargin=50, rightMargin=50, topMargin=60)
                styles = getSampleStyleSheet()
                normal = styles["Normal"]
                h1 = ParagraphStyle("H1", fontSize=20, fontName="Helvetica-Bold",
                                    spaceAfter=6, textColor=colors.HexColor("#8b0000"))
                h2 = ParagraphStyle("H2", fontSize=12, fontName="Helvetica-Bold",
                                    spaceAfter=4, textColor=colors.HexColor("#333"))
                body_s = ParagraphStyle("Body", fontSize=10, fontName="Helvetica",
                                        spaceAfter=3, textColor=colors.HexColor("#222"))

                elems = [
                    Paragraph("Smart Farmer Assistant — Field Report", h1),
                    Paragraph(f"Date: {datetime.now().strftime('%d %B %Y, %H:%M')}", body_s),
                    Paragraph(f"Location: {st.session_state.city}", body_s),
                    Spacer(1, 12),
                    Paragraph("Weather Conditions", h2),
                    Paragraph(f"Temperature: {st.session_state.temp}°C | Humidity: {st.session_state.humidity}% | Rainfall: {st.session_state.rainfall} mm", body_s),
                    Spacer(1, 10),
                    Paragraph("Crop Recommendation", h2),
                ]
                for i,(c,p) in enumerate(top_c):
                    elems.append(Paragraph(f"#{i+1} {c}  —  AI Confidence: {p}%", body_s))

                elems += [
                    Spacer(1, 10),
                    Paragraph("Financial Summary", h2),
                ]
                tbl_data = [
                    ["Metric", "Value"],
                    ["Yield (kg)",          f"{fp.get('total_yield_kg',0):,.0f}"],
                    ["Revenue",             f"Rs {fp.get('revenue',0):,.0f}"],
                    ["Input Cost",          f"Rs {fp.get('total_input_cost',0):,.0f}"],
                    ["Net Profit",          f"Rs {fp.get('profit',0):,.0f}"],
                    ["Buy Budget",          f"Rs {fp.get('seed_fert_budget',0):,.0f}"],
                    ["Target Sell Price",   f"Rs {fp.get('target_sell_price',0):.2f}/kg"],
                ]
                tbl = Table(tbl_data, colWidths=[250, 200])
                tbl.setStyle(TableStyle([
                    ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#8b0000")),
                    ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
                    ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.HexColor("#f9f9f9"), colors.white]),
                    ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
                    ("FONTSIZE",     (0,0), (-1,-1), 9),
                    ("TOPPADDING",   (0,0), (-1,-1), 5),
                    ("BOTTOMPADDING",(0,0), (-1,-1), 5),
                ]))
                elems.append(tbl)

                # Alerts
                elems += [Spacer(1,10), Paragraph("Smart Advisories", h2)]
                for level, msg in fp.get("alerts", []):
                    elems.append(Paragraph(f"[{level.upper()}] {msg}", body_s))

                doc.build(elems)
                buf.seek(0)
                return buf.read()

            pdf_bytes = build_pdf()
            st.download_button(
                "📥  Download PDF Report",
                data=pdf_bytes,
                file_name=f"smart_farmer_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )
        else:
            st.info("Install reportlab to enable PDF export: `pip install reportlab`")

        # JSON export
        if fp:
            report_json = {
                "date":       datetime.now().isoformat(),
                "location":   st.session_state.city,
                "weather":    {"temp":st.session_state.temp,"humidity":st.session_state.humidity,"rainfall":st.session_state.rainfall},
                "top_crops":  top_c,
                "financials": fp,
            }
            st.download_button(
                "⬇  Export JSON",
                data=json.dumps(report_json, indent=2),
                file_name="smart_farmer_report.json",
                mime="application/json",
            )
