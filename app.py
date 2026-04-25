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
    """
    Load crop_data.csv (FAO schema):
      Columns: Area, Item, Year, hg/ha_yield,
               average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp
    Normalises into a flat DataFrame with consistent column names.
    """
    candidates = [
        os.path.join(BASE_DIR, "data", "crop_data.csv"),
        os.path.join(BASE_DIR, "crop_data.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]

            # ── Rename FAO columns to internal names ──────────────────
            rename = {
                "Item":                          "label",
                "hg/ha_yield":                   "yield_hg_ha",   # hectograms/ha
                "average_rain_fall_mm_per_year":  "rainfall",
                "avg_temp":                       "temperature",
                "pesticides_tonnes":              "pesticides",
                "Area":                           "area",
                "Year":                           "year",
            }
            df.rename(columns={k:v for k,v in rename.items() if k in df.columns}, inplace=True)

            # hg/ha → kg/ha  (1 hg = 0.1 kg)
            df["yield_kg_ha"] = pd.to_numeric(df["yield_hg_ha"], errors="coerce") * 0.1
            df["rainfall"]    = pd.to_numeric(df["rainfall"],    errors="coerce")
            df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
            df["pesticides"]  = pd.to_numeric(df["pesticides"],  errors="coerce").fillna(0)

            df.dropna(subset=["yield_kg_ha", "rainfall", "temperature"], inplace=True)

            # Approximate cost & price from yield (since CSV has no price column)
            # Cost ≈ yield-based heuristic per crop; price mapped from known ranges
            cost_map = {
                "Rice, paddy":         35000, "Wheat":       28000, "Maize":        30000,
                "Potatoes":            38000, "Soybeans":    25000, "Cassava":       20000,
                "Sorghum":             22000, "Sweet potatoes": 25000,
                "Plantains and others":40000, "Yams":        30000,
            }
            price_map = {
                "Rice, paddy":  22, "Wheat":   20, "Maize":         15,
                "Potatoes":     18, "Soybeans":35, "Cassava":       12,
                "Sorghum":      14, "Sweet potatoes": 20,
                "Plantains and others": 30, "Yams": 25,
            }
            df["cost_per_ha"]  = df["label"].map(cost_map).fillna(28000)
            df["price_per_kg"] = df["label"].map(price_map).fillna(20)

            return df

    # Should not reach here if crop_data.csv is in repo
    raise FileNotFoundError("crop_data.csv not found. Place it in data/ folder.")

@st.cache_data(show_spinner=False)
def build_model(df: pd.DataFrame):
    """
    Train a Random Forest crop classifier on real FAO features:
      temperature, rainfall, pesticides  →  label (crop)
    Also compute per-crop stats (mean yield, std, cost, price) from the data.
    """
    feature_cols = [c for c in ["temperature", "rainfall", "pesticides"] if c in df.columns]
    if not feature_cols or "label" not in df.columns or not SKLEARN_OK:
        return None, None, None, {}

    df_clean = df.dropna(subset=feature_cols + ["label", "yield_kg_ha"])

    le = LabelEncoder()
    y  = le.fit_transform(df_clean["label"])
    X  = df_clean[feature_cols]

    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    # ── Per-crop stats derived from actual CSV data ───────────────────
    stats = {}
    for crop, grp in df_clean.groupby("label"):
        yield_vals = grp["yield_kg_ha"]
        stats[crop] = {
            "yield_kg_ha":  yield_vals.mean(),
            "yield_std":    yield_vals.std(),
            "cost_per_ha":  grp["cost_per_ha"].mean()  if "cost_per_ha"  in grp else 28000,
            "price_per_kg": grp["price_per_kg"].mean() if "price_per_kg" in grp else 20,
            "avg_temp":     grp["temperature"].mean(),
            "avg_rain":     grp["rainfall"].mean(),
            "n_records":    len(grp),
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


def predict_crop(clf, le, feature_cols, temp, rain, pesticides, top_n=3):
    """Return top N crop predictions with probabilities using real features."""
    if clf is None:
        return []
    feat_map = {"temperature": temp, "rainfall": rain, "pesticides": pesticides}
    X = np.array([[feat_map.get(f, 0) for f in feature_cols]])
    proba   = clf.predict_proba(X)[0]
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

    # ── Weather-based inputs (auto-filled from live weather) ─────────
    st.markdown("<div class='section-title'>🌡 Climate Inputs (Auto-filled)</div>", unsafe_allow_html=True)

    temp_input = st.number_input(
        "Temperature (°C)", -10.0, 50.0,
        float(round(st.session_state.temp, 1)), step=0.5,
        help="Auto-filled from live weather"
    )
    rain_input = st.number_input(
        "Rainfall (mm/year)", 0.0, 4000.0,
        float(round(st.session_state.rainfall * 365, 0)),  # daily → annual estimate
        step=10.0,
        help="Annual rainfall estimate. Auto-seeded from live reading."
    )
    pest_input = st.number_input(
        "Pesticide Usage (tonnes)", 0.0, 500.0, 50.0, step=5.0,
        help="Typical pesticide tonnes used in your region"
    )

    st.markdown("---")

    # ── Land size ────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🌿 Farm Details</div>", unsafe_allow_html=True)
    land_acres = st.number_input("Land Size (Acres)", 0.5, 1000.0, 5.0, step=0.5)

    st.markdown("---")
    if st.button("🔮  Analyse & Predict", use_container_width=True):
        top_crops = predict_crop(
            clf, le, feature_cols,
            temp_input, rain_input, pest_input,
            top_n=3
        )
        if not top_crops and stats:
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

        # ── Live Crop Comparison Table ────────────────────────────────
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        section_title("📡 LIVE CROP COMPARISON — ALL CROPS FOR CURRENT WEATHER")

        if stats:
            import plotly.express as px
            import plotly.graph_objects as go

            ha_val = land_acres * 0.4047

            # Build comparison rows for every crop using current weather
            comparison_rows = []
            for crop_name, s in stats.items():
                # Same weather factor logic as financial_advisory
                wf = 1.0
                t  = st.session_state.temp
                r  = st.session_state.rainfall
                h  = st.session_state.humidity if hasattr(st.session_state, "humidity") else 60

                if   t > 38: wf *= 0.72
                elif t > 34: wf *= 0.88
                elif t < 12: wf *= 0.80

                if   h > 85: wf *= 0.90
                elif h < 30: wf *= 0.85

                if   r > 15: wf *= 0.85
                elif r > 5:  wf *= 1.05
                elif r < 1:  wf *= 0.88

                adj_yield   = s["yield_kg_ha"] * wf
                total_yield = adj_yield * ha_val
                total_cost  = s["cost_per_ha"] * ha_val
                revenue     = total_yield * s["price_per_kg"]
                profit      = revenue - total_cost
                cv          = round(s["yield_std"] / max(s["yield_kg_ha"], 1) * 100, 1)
                target_sell = (total_cost * 1.20) / max(total_yield, 1)

                # Risk tier
                if   cv <= 15: risk = "🟢 Low"
                elif cv <= 30: risk = "🟡 Medium"
                else:          risk = "🔴 High"

                # Rank badge
                is_primary   = crop_name == st.session_state.primary_crop
                is_secondary = crop_name == st.session_state.secondary_crop

                comparison_rows.append({
                    "Crop":               crop_name,
                    "AI Pick":            "⭐ #1" if is_primary else ("🥈 #2" if is_secondary else ""),
                    "Adj. Yield/ha (kg)": round(adj_yield, 0),
                    "Total Yield (kg)":   round(total_yield, 0),
                    "Revenue (₹)":        round(revenue, 0),
                    "Input Cost (₹)":     round(total_cost, 0),
                    "Net Profit (₹)":     round(profit, 0),
                    "Sell Price (₹/kg)":  round(target_sell, 2),
                    "CV (%)":             cv,
                    "Risk":               risk,
                    "Wx Factor":          round(wf, 2),
                })

            df_compare = pd.DataFrame(comparison_rows).sort_values("Net Profit (₹)", ascending=False).reset_index(drop=True)

            # ── Bar chart — Net Profit comparison ─────────────────────
            DARK_C = dict(plot_bgcolor="#0d0d0d", paper_bgcolor="#0d0d0d",
                          font_color="#e8e8e8", font_family="EB Garamond")

            bar_colors = [
                "#c0392b" if r["AI Pick"] == "⭐ #1"
                else "#9b59b6" if r["AI Pick"] == "🥈 #2"
                else "#2a2a2a"
                for _, r in df_compare.iterrows()
            ]

            fig_bar = go.Figure(go.Bar(
                x=df_compare["Crop"],
                y=df_compare["Net Profit (₹)"],
                marker_color=bar_colors,
                text=[f"₹{v:,.0f}" for v in df_compare["Net Profit (₹)"]],
                textposition="outside",
                textfont=dict(size=10, color="#e8e8e8"),
            ))
            fig_bar.update_layout(
                **DARK_C,
                title="Net Profit Comparison Across All Crops — Current Weather & Land Size",
                title_font_size=13,
                xaxis_title="", yaxis_title="Net Profit (₹)",
                margin=dict(l=10, r=10, t=50, b=10),
                showlegend=False,
            )
            # Highlight zero line
            fig_bar.add_hline(y=0, line_color="#5c0a7d", line_dash="dot", line_width=1)
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Scatter — Profit vs Risk ───────────────────────────────
            fig_sc = px.scatter(
                df_compare, x="CV (%)", y="Net Profit (₹)",
                size="Total Yield (kg)", color="Risk", text="Crop",
                color_discrete_map={"🟢 Low":"#2ecc71","🟡 Medium":"#f39c12","🔴 High":"#e74c3c"},
                title="Profit vs Risk Matrix — Bubble size = Total Yield",
            )
            fig_sc.update_traces(textposition="top center", textfont_size=10,
                                 marker_line_color="#0d0d0d", marker_line_width=0.5)
            fig_sc.update_layout(**DARK_C, title_font_size=13,
                                 margin=dict(l=10, r=10, t=50, b=10))
            fig_sc.add_hline(y=0, line_color="#8b0000", line_dash="dot", line_width=1)
            st.plotly_chart(fig_sc, use_container_width=True)

            # ── Full comparison table ─────────────────────────────────
            st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
            noir_card(f"""
            <div class='unit-label'>TABLE KEY</div>
            <span style='color:#c0392b'>■</span> <b>⭐ AI #1 Pick</b> &nbsp;&nbsp;
            <span style='color:#9b59b6'>■</span> <b>🥈 AI #2 Pick</b> &nbsp;&nbsp;
            Sell Price guarantees <b>20% margin</b> over input cost &nbsp;&nbsp;
            Wx Factor = weather adjustment applied to yield
            """, accent="gold")

            def style_row(row):
                if row["AI Pick"] == "⭐ #1":
                    return ["background-color: rgba(139,0,0,0.18); color:#e8e8e8"] * len(row)
                elif row["AI Pick"] == "🥈 #2":
                    return ["background-color: rgba(92,10,125,0.18); color:#e8e8e8"] * len(row)
                return ["color:#e8e8e8"] * len(row)

            st.dataframe(
                df_compare.style
                    .apply(style_row, axis=1)
                    .format({
                        "Adj. Yield/ha (kg)": "{:,.0f}",
                        "Total Yield (kg)":   "{:,.0f}",
                        "Revenue (₹)":        "{:,.0f}",
                        "Input Cost (₹)":     "{:,.0f}",
                        "Net Profit (₹)":     "{:,.0f}",
                        "Sell Price (₹/kg)":  "{:.2f}",
                        "CV (%)":             "{:.1f}",
                        "Wx Factor":          "{:.2f}",
                    }),
                use_container_width=True,
                hide_index=True,
            )


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
            palette = ["#8b0000","#5c0a7d","#c9a84c","#2ecc71","#e74c3c","#9b59b6"]
            fig = go.Figure(go.Bar(
                x=imp, y=feature_cols, orientation="h",
                marker_color=palette[:len(feature_cols)],
                text=[f"{v:.3f}" for v in imp],
                textposition="outside",
            ))
            fig.update_layout(**DARK_TPL, title="🧠 Feature Importance — What drives crop recommendation",
                              title_font_size=13, margin=dict(l=10,r=60,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)
    else:
        noir_card("<i>Analytics unavailable — load crop_data.csv to unlock.</i>")


# ── TAB 3 — RISK ORACLE ─────────────────────────────────────────────
with tab3:
    import plotly.express as px
    import plotly.graph_objects as go

    section_title("RISK ORACLE")

    # Build df_risk directly from the real loaded DataFrame
    grp = df.groupby("label")["yield_kg_ha"].agg(["mean","std","count"]).reset_index()
    grp.columns = ["Crop", "Mean Yield (kg/ha)", "Std Dev", "Records"]
    grp["Std Dev"]  = grp["Std Dev"].fillna(0)
    grp["CV (%)"]   = (grp["Std Dev"] / grp["Mean Yield (kg/ha)"].replace(0, np.nan) * 100).fillna(0).round(1)

    # Avg temp & rain per crop
    climate_grp = df.groupby("label")[["temperature","rainfall"]].mean().reset_index()
    climate_grp.columns = ["Crop","Avg Temp (°C)","Avg Rain (mm)"]
    df_risk = grp.merge(climate_grp, on="Crop", how="left")

    # Cost & price columns (from mapped values in df)
    cost_grp = df.groupby("label")[["cost_per_ha","price_per_kg"]].mean().reset_index()
    cost_grp.columns = ["Crop","Cost/ha (₹)","Price/kg (₹)"]
    df_risk = df_risk.merge(cost_grp, on="Crop", how="left")

    # Risk tier using dynamic quantiles
    q33 = df_risk["CV (%)"].quantile(0.33)
    q66 = df_risk["CV (%)"].quantile(0.66)
    def risk_tier(cv):
        if cv <= q33:   return "🟢 Low Risk"
        elif cv <= q66: return "🟡 Medium Risk"
        return "🔴 High Risk"
    df_risk["Risk Tier"] = df_risk["CV (%)"].apply(risk_tier)

    DARK_TPL2 = dict(plot_bgcolor="#0d0d0d", paper_bgcolor="#0d0d0d",
                     font_color="#e8e8e8", font_family="EB Garamond")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(df_risk, names="Risk Tier",
                     color="Risk Tier",
                     color_discrete_map={
                         "🟢 Low Risk":    "#2ecc71",
                         "🟡 Medium Risk": "#f39c12",
                         "🔴 High Risk":   "#e74c3c",
                     },
                     title="Yield Variability Risk Distribution", hole=0.45)
        fig.update_layout(**DARK_TPL2, title_font_size=13,
                          margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df_risk.sort_values("CV (%)", ascending=False),
            x="Crop", y="CV (%)", color="Risk Tier",
            color_discrete_map={
                "🟢 Low Risk":    "#2ecc71",
                "🟡 Medium Risk": "#f39c12",
                "🔴 High Risk":   "#e74c3c",
            },
            title="Coefficient of Variation (%) — Yield Volatility by Crop",
        )
        fig.update_layout(**DARK_TPL2, title_font_size=13,
                          margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Yield mean vs std scatter
    fig = px.scatter(
        df_risk, x="Mean Yield (kg/ha)", y="Std Dev",
        size="CV (%)", color="Risk Tier", text="Crop",
        color_discrete_map={
            "🟢 Low Risk":"#2ecc71","🟡 Medium Risk":"#f39c12","🔴 High Risk":"#e74c3c"
        },
        title="Mean Yield vs Yield Volatility — Risk Matrix",
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(**DARK_TPL2, title_font_size=13,
                      margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Full risk table
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    section_title("FULL RISK TABLE — DERIVED FROM CROP DATA CSV")
    show_cols = ["Crop","Records","Mean Yield (kg/ha)","Std Dev","CV (%)","Risk Tier",
                 "Avg Temp (°C)","Avg Rain (mm)","Cost/ha (₹)","Price/kg (₹)"]
    show_df = df_risk[show_cols].sort_values("CV (%)").reset_index(drop=True)
    st.dataframe(
        show_df.style.background_gradient(subset=["CV (%)"], cmap="RdYlGn_r")
                     .format({"Mean Yield (kg/ha)":"{:,.0f}","Std Dev":"{:,.0f}",
                              "CV (%)":"{:.1f}","Avg Temp (°C)":"{:.1f}",
                              "Avg Rain (mm)":"{:.0f}","Cost/ha (₹)":"{:,.0f}",
                              "Price/kg (₹)":"{:.0f}"}),
        use_container_width=True, hide_index=True
    )


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
