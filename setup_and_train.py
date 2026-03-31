"""
===============================================================
  Smart Crop Profit & Risk Prediction System
  MODULE: Setup Script (Run this FIRST before the app)
  Description: One-click script that:
    1. Generates the dataset
    2. Runs preprocessing
    3. Trains the ML model
    4. Saves all artifacts to disk

  Usage:
    python setup_and_train.py
===============================================================
"""

import os
import sys

# Ensure the project root is in the Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

print("=" * 60)
print("  SMART CROP PROFIT & RISK PREDICTION SYSTEM")
print("  One-Click Setup & Training Script")
print("=" * 60)

# ─────────────────────────────
# STEP 1: Generate Dataset
# ─────────────────────────────
print("\n[1/4] Generating dataset...")
from data.generate_dataset import df_raw, df_summary
df_raw.to_csv(os.path.join(BASE_DIR, "data", "crop_raw_data.csv"), index=False)
df_summary.to_csv(os.path.join(BASE_DIR, "data", "crop_summary.csv"), index=False)
print("      ✅ Dataset saved to data/ folder")

# ─────────────────────────────
# STEP 2: Preprocessing
# ─────────────────────────────
print("\n[2/4] Running preprocessing...")
from data.preprocessing import run_preprocessing, FEATURE_COLS, TARGET_COL, scale_features
from sklearn.model_selection import train_test_split

raw_path = os.path.join(BASE_DIR, "data", "crop_raw_data.csv")
summary_path = os.path.join(BASE_DIR, "data", "crop_summary.csv")

df_processed, df_summary_loaded, label_encoder = run_preprocessing(raw_path, summary_path)
print("      ✅ Preprocessing complete")

# ─────────────────────────────
# STEP 3: Train Model
# ─────────────────────────────
print("\n[3/4] Training models...")
from model_training import train_and_evaluate, save_artifacts

X = df_processed[FEATURE_COLS]
y = df_processed[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

best_model, best_name, results = train_and_evaluate(X_train, X_test, y_train, y_test, scaler)

# Retrain best model on FULL data for production use
X_all_sc, final_scaler = scale_features(X)
best_model.fit(X_all_sc, y)
print(f"\n      ✅ Best model: {best_name} — retrained on full dataset")

# ─────────────────────────────
# STEP 4: Save Artifacts
# ─────────────────────────────
print("\n[4/4] Saving model artifacts...")
models_dir = os.path.join(BASE_DIR, "models")
save_artifacts(best_model, final_scaler, label_encoder, models_dir)

# Also annotate and save the final summary
from utils.risk_analysis import annotate_summary
df_annotated = annotate_summary(df_summary_loaded)
df_annotated.to_csv(os.path.join(BASE_DIR, "data", "crop_annotated.csv"), index=False)
print("      ✅ Annotated summary saved to data/crop_annotated.csv")

print("\n" + "=" * 60)
print("  ✅ SETUP COMPLETE! You can now run the app:")
print("     streamlit run app.py")
print("=" * 60)
