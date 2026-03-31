# 🌾 Smart Crop Profit & Risk Prediction System

**Mini Project — Department of AI & Data Science**  
**Sikkim Manipal Institute of Technology, SMU**

**Team:** Agrim Singh (202300550) · Alisha Dhakal (202300288) · Rajdeep Dey (202300325)  
**Supervisor:** Mr. Gaurav Sarma

---

## 📋 Project Overview

A data-driven decision support system that helps Indian farmers decide **which crop to grow** by analyzing:

- 💰 **Expected Profit** — calculated from yield, market price, and cultivation cost
- ⚠️ **Financial Risk** — measured using price standard deviation and coefficient of variation
- 📌 **Recommendations** — personalized advice per crop selection

---

## 🗂️ Folder Structure

```
smart_crop_project/
│
├── app.py                    ← Main Streamlit dashboard (run this)
├── setup_and_train.py        ← One-click setup: generate data + train model
├── model_training.py         ← ML model training, evaluation, save/load
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
│
├── data/
│   ├── generate_dataset.py   ← Synthetic Indian crop dataset generator
│   ├── preprocessing.py      ← Data cleaning, feature engineering, encoding
│   ├── crop_raw_data.csv     ← Generated raw dataset (5 years × 15 crops)
│   ├── crop_summary.csv      ← Per-crop aggregated stats
│   └── crop_annotated.csv    ← Final dataset with recommendations & ranks
│
├── models/
│   ├── crop_profit_model.pkl ← Trained Random Forest model
│   ├── scaler.pkl            ← StandardScaler for feature normalization
│   └── label_encoder.pkl     ← LabelEncoder for crop name encoding
│
└── utils/
    ├── risk_analysis.py      ← Risk classification & recommendation logic
    └── visualization.py      ← All Plotly chart functions
```

---

## 🚀 How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate data & train the model (run ONCE)
```bash
python setup_and_train.py
```

### Step 3: Launch the Streamlit app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🔬 Technical Details

### Dataset
- 15 Indian crops (Rice, Wheat, Cotton, Tomato, Onion, etc.)
- 5 years of synthetic price data per crop (2019–2023)
- Features: crop name, cost/ha, yield/ha, market price, derived profit

### Profit Formula
```
Profit = (Yield × Market Price) − Cost of Cultivation
```

### Risk Metrics
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Standard Deviation | σ of price across years | Absolute price spread |
| Coefficient of Variation (CV) | (σ / mean) × 100 | % price variability |

### Risk Classification
| CV (%) | Risk Level |
|--------|------------|
| < 10   | 🟢 Low Risk |
| 10–20  | 🟡 Medium Risk |
| ≥ 20   | 🔴 High Risk |

### ML Model
- **Algorithm:** Random Forest Regressor (200 trees)
- **Target:** Profit per hectare
- **Features:** Crop (encoded), cost, yield, market price, revenue, price-to-cost ratio
- **Comparison:** Also evaluates Linear Regression and Gradient Boosting
- **Best model** is selected by R² score and saved to disk

### Recommendation Engine
The system combines profit tier (High/Moderate/Low based on percentile rank) with risk level to generate one of 9 recommendation messages:

| Profit + Risk | Recommendation |
|---------------|----------------|
| High + Low | ✅ BEST CHOICE — Strongly Recommended |
| High + High | ⚠️ Proceed with caution |
| Moderate + Low | 🌿 Safe, stable choice |
| Low + High | 🚫 Strongly NOT Recommended |
| ... | ... |

### Safety Score (Bonus Feature)
```
Safety Score = Avg Profit ÷ (1 + CV)
```
Higher score = better profit-to-risk ratio. Used to rank all crops and identify the single **Safest Crop**.

---

## 📊 App Features

| Tab | Contents |
|-----|----------|
| 📊 Prediction Dashboard | Per-crop profit prediction, risk level, recommendation, historical trend |
| 📈 Profit Comparison | Horizontal bar chart of all crops + safety ranking |
| ⚠️ Risk Analysis | CV bar chart + Profit vs Risk scatter plot with quadrants |
| 📋 Data Table | Full annotated table with download button |

---

## 📚 References
1. Thapaswi & Gunashekar, IEEE TENSYMP 2024 — *Predicting Crop Prices Using ML*
2. Prabhakar et al., Biology & Life Sciences Forum 2025 — *ML-Based Agricultural Price Forecasting*
3. Kumar & Singh, IJEAST 2024 — *Crop Price Prediction Using Machine Learning*
