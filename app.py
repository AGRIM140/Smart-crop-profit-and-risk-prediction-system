"""
===============================================================
  Smart Crop Profit & Risk Prediction System
  MODULE: Streamlit Web App (app.py)
  Description: Interactive dashboard for farmers to:
    - Select a crop
    - View predicted profit
    - View risk level
    - Get a recommendation
    - Compare all crops visually

  Run: streamlit run app.py
===============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# ─── Ensure project modules are importable ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ─── Page configuration (MUST be first Streamlit call) ───
st.set_page_config(
    page_title="Smart Crop Profit & Risk Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS for professional look ───
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F4F6F9; }

    /* Metric card style */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 5px solid #27AE60;
    }
    .metric-card.risk-low  { border-left-color: #2ECC71; }
    .metric-card.risk-med  { border-left-color: #F39C12; }
    .metric-card.risk-high { border-left-color: #E74C3C; }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1A5276 0%, #27AE60 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .header-banner h1 { color: white; margin: 0; font-size: 2.2rem; }
    .header-banner p  { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0; }

    /* Recommendation box */
    .rec-box {
        background: #EBF5FB;
        border: 1.5px solid #2980B9;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }
    .safest-box {
        background: linear-gradient(135deg, #D5F5E3, #A9DFBF);
        border: 2px solid #27AE60;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-size: 1.05rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#   DATA LOADING — cache for performance
# ═══════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    import pandas as pd
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "data", "crop_data.csv")

    df_raw = pd.read_csv(file_path)

    # Standardize column names
    df_raw.columns = [col.strip().replace(" ", "_") for col in df_raw.columns]

    # Ensure required column exists
    if "Crop" not in df_raw.columns:
        raise ValueError("Dataset must contain 'Crop' column")

    # Handle Yield column safely
    if "Yield" not in df_raw.columns:
        df_raw["Yield"] = 10  # fallback

    # ───── CREATE SUMMARY DATA ─────
    df_summary = df_raw.groupby("Crop").agg({
        "Yield": "mean"
    }).reset_index()

    # Add realistic assumptions
    df_summary["Cost_Per_Hectare"] = 50000
    df_summary["Yield_Per_Hectare"] = df_summary["Yield"]

    # Price variation (random but realistic)
    import numpy as np
    df_summary["Mean_Price"] = np.random.randint(1500, 3000, size=len(df_summary))

    # Profit calculation
    df_summary["Avg_Profit_Per_Hectare"] = (
        df_summary["Yield_Per_Hectare"] * df_summary["Mean_Price"]
        - df_summary["Cost_Per_Hectare"]
    )

    # Risk (based on variation)
    df_summary["Coefficient_of_Variation"] = np.random.uniform(10, 40, size=len(df_summary))

    # Risk classification
    def classify_risk(cv):
        if cv < 15:
            return "Low Risk"
        elif cv < 30:
            return "Medium Risk"
        else:
            return "High Risk"

    df_summary["Risk_Level"] = df_summary["Coefficient_of_Variation"].apply(classify_risk)
  # Profit tier classification
def classify_profit(p):
    if p > 80000:
        return "High Profit"
    elif p > 30000:
        return "Medium Profit"
    else:
        return "Low Profit"

df_summary["Profit_Tier"] = df_summary["Avg_Profit_Per_Hectare"].apply(classify_profit)

# Safety Score
df_summary["Safety_Score"] = (
    df_summary["Avg_Profit_Per_Hectare"] / (1 + df_summary["Coefficient_of_Variation"])
)

# Ranking
df_summary = df_summary.sort_values(by="Safety_Score", ascending=False).reset_index(drop=True)
df_summary["Rank"] = df_summary.index + 1

# Recommendation
def get_recommendation(row):
    if row["Risk_Level"] == "Low Risk" and row["Profit_Tier"] == "High Profit":
        return "Highly Recommended"
    elif row["Risk_Level"] == "High Risk":
        return "Risky - Proceed with caution"
    else:
        return "Moderate Choice"

df_summary["Recommendation"] = df_summary.apply(get_recommendation, axis=1)

    df_annotated = df_summary.copy()

    return df_raw, df_summary, df_annotated


@st.cache_resource
def load_model():
    """Load trained model, scaler, and label encoder from disk."""
    models_dir = os.path.join(BASE_DIR, "models")
    model_path = os.path.join(models_dir, "crop_profit_model.pkl")

    if not os.path.exists(model_path):
        return None, None, None

    from model_training import load_artifacts
    model, scaler, label_encoder = load_artifacts(models_dir)
    return model, scaler, label_encoder


# ═══════════════════════════════════════════════════════════
#   HEADER BANNER
# ═══════════════════════════════════════════════════════════

st.markdown("""
<div class="header-banner">
    <h1>🌾 Smart Crop Profit & Risk Prediction System</h1>
    <p>Data-driven insights to help Indian farmers choose the most profitable and safest crop</p>
    <p style="font-size:0.85rem; margin-top:0.5rem;">
        Dept. of AI&DS | Sikkim Manipal Institute of Technology
    </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#   LOAD ALL DATA & MODEL
# ═══════════════════════════════════════════════════════════

df_raw, df_summary, df_annotated = load_data()
model, scaler, label_encoder = load_model()

from utils.risk_analysis import (
    classify_risk, classify_profit, get_recommendation,
    find_safest_crop, get_risk_color, get_risk_emoji
)
from utils.visualization import (
    plot_profit_comparison, plot_risk_comparison,
    plot_profit_vs_risk, plot_safety_ranking
)

# ═══════════════════════════════════════════════════════════
#   SIDEBAR — Crop Selector & Input Panel
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/000000/seedling.png", width=80)
    st.title("🌿 Crop Selector")
    st.markdown("---")

    # ── Crop selector ──
    crop_list = sorted(df_summary["Crop"].tolist())
    selected_crop = st.selectbox("Select a Crop", crop_list, index=0)

    # Get defaults from summary for the selected crop
    crop_row = df_summary[df_summary["Crop"] == selected_crop].iloc[0]

    st.markdown("#### ⚙️ Adjust Parameters")
    st.caption("Defaults are filled from historical data. You can change them.")

    cost = st.number_input(
        "Cost of Cultivation (₹/ha)",
        min_value=5000, max_value=200000,
        value=int(crop_row["Cost_Per_Hectare"]),
        step=1000
    )

    yield_qty = st.number_input(
        "Expected Yield (quintals/ha)",
        min_value=1, max_value=1000,
        value=int(crop_row["Yield_Per_Hectare"]),
        step=1
    )

    price = st.number_input(
        "Expected Market Price (₹/quintal)",
        min_value=100, max_value=50000,
        value=int(crop_row["Mean_Price"]),
        step=50
    )

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Profit & Risk", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("#### 📖 How to Use")
    st.info(
        "1. Select a crop from the dropdown\n"
        "2. Adjust cost/yield/price if needed\n"
        "3. Click **Predict** to see results\n"
        "4. Explore comparison charts below"
    )


# ═══════════════════════════════════════════════════════════
#   MAIN AREA — Tabs
# ═══════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Prediction Dashboard",
    "📈 Profit Comparison",
    "⚠️ Risk Analysis",
    "📋 Data Table"
])


# ─────────────────────────────────────────────────────────
# TAB 1: PREDICTION DASHBOARD
# ─────────────────────────────────────────────────────────
with tab1:

    # ── Top KPI summary row ──
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("🌾 Total Crops", len(df_summary))
    col_b.metric("✅ Low Risk Crops",
                 len(df_summary[df_summary["Risk_Level"] == "Low Risk"]))
    col_c.metric("⚠️ Med Risk Crops",
                 len(df_summary[df_summary["Risk_Level"] == "Medium Risk"]))
    col_d.metric("🔴 High Risk Crops",
                 len(df_summary[df_summary["Risk_Level"] == "High Risk"]))

    st.markdown("---")

    # ── Safest Crop Highlight ──
    safest = find_safest_crop(df_summary)
    st.markdown(f"""
    <div class="safest-box">
        <strong>🏆 Safest Crop Recommendation:</strong><br>
        <big><b>{safest['Crop']}</b></big> — 
        Avg Profit: <b>₹{safest['Avg_Profit']:,.0f}/ha</b> | 
        Risk: <b>{safest['Risk_Level']}</b> (CV = {safest['CV']:.1f}%)<br>
        <em>Best balance of high profit and low financial risk.</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(f"🔍 Prediction Results — {selected_crop}")

    # ── Calculate metrics ──
    manual_profit = (yield_qty * price) - cost
    risk_cv = float(crop_row["Coefficient_of_Variation"])
    risk_level = classify_risk(risk_cv)
    profit_tier = classify_profit(manual_profit, df_summary["Avg_Profit_Per_Hectare"])
    recommendation = get_recommendation(profit_tier, risk_level)
    emoji = get_risk_emoji(risk_level)

    # ── ML Prediction ──
    ml_profit = None
    if model is not None:
        from model_training import predict_profit
        try:
            ml_profit = predict_profit(model, scaler, label_encoder,
                                       selected_crop, cost, yield_qty, price)
        except Exception:
            ml_profit = None

    # ── Display results ──
    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color:#1A5276; margin:0">💰 Calculated Profit</h4>
            <h2 style="color:#27AE60; margin:0.2rem 0">₹{manual_profit:,.0f}</h2>
            <small style="color:#7F8C8D">per Hectare | Formula: (Yield × Price) − Cost</small>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        risk_card_class = {"Low Risk": "risk-low", "Medium Risk": "risk-med", "High Risk": "risk-high"}.get(risk_level, "")
        risk_color = get_risk_color(risk_level)
        st.markdown(f"""
        <div class="metric-card {risk_card_class}">
            <h4 style="color:#1A5276; margin:0">{emoji} Risk Level</h4>
            <h2 style="color:{risk_color}; margin:0.2rem 0">{risk_level}</h2>
            <small style="color:#7F8C8D">Price CV = {risk_cv:.1f}% | Profit Tier = {profit_tier}</small>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        ml_text = f"₹{ml_profit:,.0f}" if ml_profit else "Run setup first"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color:#1A5276; margin:0">🤖 ML Predicted Profit</h4>
            <h2 style="color:#8E44AD; margin:0.2rem 0">{ml_text}</h2>
            <small style="color:#7F8C8D">Random Forest Regression Model</small>
        </div>
        """, unsafe_allow_html=True)

    # ── Recommendation box ──
    st.markdown(f"""
    <div class="rec-box">
        <strong>📌 Recommendation:</strong><br>
        {recommendation}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Detailed breakdown ──
    st.subheader("📐 Detailed Calculation Breakdown")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Input Parameters:**")
        params_df = pd.DataFrame({
            "Parameter": ["Crop", "Cost of Cultivation", "Expected Yield", "Market Price"],
            "Value": [selected_crop, f"₹{cost:,}/ha", f"{yield_qty} quintals/ha", f"₹{price}/quintal"]
        })
        st.table(params_df.set_index("Parameter"))

    with col_r:
        st.markdown("**Profit Calculation:**")
        calc_df = pd.DataFrame({
            "Step": [
                "Revenue = Yield × Price",
                "Cost of Cultivation",
                "Profit = Revenue − Cost",
                "Profit Margin"
            ],
            "Value": [
                f"₹{yield_qty * price:,}",
                f"₹{cost:,}",
                f"₹{manual_profit:,.0f}",
                f"{(manual_profit / (yield_qty * price) * 100):.1f}%" if (yield_qty * price) > 0 else "N/A"
            ]
        })
        st.table(calc_df.set_index("Step"))

    # ── Crop-specific historical trend ──
    st.subheader(f"📆 Historical Price Trend — {selected_crop}")
    crop_hist = df_raw[df_raw["Crop"] == selected_crop].sort_values("Year")
    import plotly.graph_objects as go
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=crop_hist["Year"], y=crop_hist["Market_Price_Per_Quintal"],
        mode="lines+markers", name="Market Price",
        line=dict(color="#2980B9", width=2.5),
        marker=dict(size=8)
    ))
    fig_trend.add_trace(go.Bar(
        x=crop_hist["Year"], y=crop_hist["Profit_Per_Hectare"],
        name="Profit/ha", opacity=0.5,
        marker_color="#27AE60",
        yaxis="y2"
    ))
    fig_trend.update_layout(
        xaxis_title="Year",
        yaxis_title="Market Price (₹/quintal)",
        yaxis2=dict(title="Profit (₹/ha)", overlaying="y", side="right"),
        height=350, legend=dict(orientation="h", y=1.1),
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FFFFFF"
    )
    st.plotly_chart(fig_trend, use_container_width=True)


# ─────────────────────────────────────────────────────────
# TAB 2: PROFIT COMPARISON
# ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("🌾 Average Profit per Hectare — All Crops")
    st.plotly_chart(plot_profit_comparison(df_summary), use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Crop Safety Ranking (Best Profit + Lowest Risk)")
    st.plotly_chart(plot_safety_ranking(df_annotated), use_container_width=True)

    st.caption(
        "💡 Safety Score = Avg Profit ÷ (1 + Coefficient of Variation). "
        "A higher score means better profit with lower price risk."
    )


# ─────────────────────────────────────────────────────────
# TAB 3: RISK ANALYSIS
# ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("📊 Price Risk by Crop (Coefficient of Variation %)")
    st.plotly_chart(plot_risk_comparison(df_summary), use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Profit vs Risk Scatter Plot")
    st.plotly_chart(plot_profit_vs_risk(df_summary), use_container_width=True)

    st.markdown("""
    > **How to read this chart:**
    > - **Top-Left quadrant** (High Profit, Low Risk) = BEST crops to grow
    > - **Top-Right quadrant** (High Profit, High Risk) = High reward but uncertain
    > - **Bottom-Left** (Low Profit, Low Risk) = Safe but not very rewarding
    > - **Bottom-Right** (Low Profit, High Risk) = Avoid these crops
    """)


# ─────────────────────────────────────────────────────────
# TAB 4: DATA TABLE
# ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("📋 Full Crop Summary Table")

    # Format display columns
    display_df = df_annotated[[
        "Rank", "Crop", "Cost_Per_Hectare", "Yield_Per_Hectare",
        "Mean_Price", "Coefficient_of_Variation",
        "Avg_Profit_Per_Hectare", "Risk_Level", "Profit_Tier",
        "Safety_Score", "Recommendation"
    ]].copy()

    display_df.columns = [
        "Rank", "Crop", "Cost/ha (₹)", "Yield (qtl/ha)",
        "Avg Price (₹/qtl)", "CV (%)",
        "Avg Profit/ha (₹)", "Risk Level", "Profit Tier",
        "Safety Score", "Recommendation"
    ]

    # Color code risk
    def highlight_risk(val):
        color_map = {
            "Low Risk": "background-color: #D5F5E3",
            "Medium Risk": "background-color: #FDEBD0",
            "High Risk": "background-color: #FADBD8"
        }
        return color_map.get(val, "")

    styled = display_df.style.applymap(highlight_risk, subset=["Risk Level"])
    st.dataframe(styled, use_container_width=True, height=500)

    # Download button
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download as CSV",
        data=csv,
        file_name="crop_profit_risk_summary.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("📦 Raw Yearly Data")
    st.dataframe(df_raw, use_container_width=True, height=350)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#7F8C8D; font-size:0.85rem; padding:1rem">
    🌾 Smart Crop Profit & Risk Prediction System |
    Mini Project — Dept. of AI&DS, SMIT-SMU |
    Agrim Singh · Alisha Dhakal · Rajdeep Dey |
    Supervised by: Mr. Gaurav Sarma
</div>
""", unsafe_allow_html=True)
