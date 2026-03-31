"""
===============================================================
  Smart Crop Profit & Risk Prediction System
  MODULE: Model Training & Prediction
  Description: Trains a Random Forest Regression model to
               predict crop profit. Evaluates and saves the
               trained model for use in the Streamlit app.
===============================================================
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import our preprocessing utilities
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.preprocessing import FEATURE_COLS, TARGET_COL, scale_features


# ─────────────────────────────────────────────────────────────────────
# STEP 1: Split data into train / test sets
# ─────────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Split dataset into features and target, then train/test."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"✅ Data split — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────
# STEP 2: Define candidate models for comparison
# ─────────────────────────────────────────────────────────────────────
def get_models():
    """Return a dict of candidate regression models."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
    }


# ─────────────────────────────────────────────────────────────────────
# STEP 3: Train all models and evaluate performance
# ─────────────────────────────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, scaler):
    """Train models, print metrics, and return the best one."""
    X_train_sc, X_test_sc, _ = scale_features(X_train, X_test)
    # Use the scaler already fitted above

    models = get_models()
    results = {}

    print("\n" + "="*55)
    print("  MODEL COMPARISON RESULTS")
    print("="*55)

    for name, model in models.items():
        # Train
        model.fit(X_train_sc, y_train)

        # Predict on test set
        y_pred = model.predict(X_test_sc)

        # Metrics
        mae  = mean_absolute_error(y_train, model.predict(X_train_sc))  # train MAE
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)

        results[name] = {
            "model": model,
            "R2": round(r2, 4),
            "RMSE": round(rmse, 2),
            "MAE": round(test_mae, 2)
        }

        print(f"\n  {name}")
        print(f"    R²   : {r2:.4f}  (1.0 = perfect)")
        print(f"    RMSE : ₹{rmse:,.2f}")
        print(f"    MAE  : ₹{test_mae:,.2f}")

    # ─── Pick best model by highest R² score ───
    best_name = max(results, key=lambda k: results[k]["R2"])
    best_model = results[best_name]["model"]
    print(f"\n🏆 Best Model: {best_name} (R² = {results[best_name]['R2']})")

    return best_model, best_name, results


# ─────────────────────────────────────────────────────────────────────
# STEP 4: Save trained model and scaler using pickle
# ─────────────────────────────────────────────────────────────────────
def save_artifacts(model, scaler, label_encoder, save_dir: str):
    """Persist model, scaler, and label encoder to disk."""
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "crop_profit_model.pkl")
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    encoder_path = os.path.join(save_dir, "label_encoder.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"\n✅ Model saved to  : {model_path}")
    print(f"✅ Scaler saved to : {scaler_path}")
    print(f"✅ Encoder saved to: {encoder_path}")


# ─────────────────────────────────────────────────────────────────────
# STEP 5: Load saved artifacts (used by Streamlit app)
# ─────────────────────────────────────────────────────────────────────
def load_artifacts(save_dir: str):
    """Load and return model, scaler, and label encoder."""
    model_path = os.path.join(save_dir, "crop_profit_model.pkl")
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    encoder_path = os.path.join(save_dir, "label_encoder.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    return model, scaler, label_encoder


# ─────────────────────────────────────────────────────────────────────
# STEP 6: Predict profit for a single crop input
# ─────────────────────────────────────────────────────────────────────
def predict_profit(model, scaler, label_encoder, crop_name: str,
                   cost: float, yield_qty: float, price: float) -> float:
    """
    Predict profit for given crop parameters.
    Returns predicted profit in INR per hectare.
    """
    # Encode crop name
    crop_encoded = label_encoder.transform([crop_name])[0]

    # Derive engineered features (same as preprocessing step)
    revenue = yield_qty * price
    price_to_cost_ratio = price / cost if cost > 0 else 0

    # Build input array in the same feature order
    X_input = np.array([[
        crop_encoded,
        cost,
        yield_qty,
        price,
        revenue,
        price_to_cost_ratio
    ]])

    # Scale input
    X_scaled = scaler.transform(X_input)

    # Predict
    predicted_profit = model.predict(X_scaled)[0]
    return round(predicted_profit, 2)


# ─────────────────────────────────────────────────────────────────────
# Main entry point — run full training pipeline
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")

    raw_path = os.path.join(data_dir, "crop_raw_data.csv")
    summary_path = os.path.join(data_dir, "crop_summary.csv")

    # Import preprocessing
    sys.path.insert(0, base_dir)
    from data.preprocessing import run_preprocessing, scale_features

    print("="*55)
    print("  SMART CROP PROFIT & RISK PREDICTION SYSTEM")
    print("  Model Training Pipeline")
    print("="*55)

    # Preprocess
    df_processed, df_summary, label_encoder = run_preprocessing(raw_path, summary_path)

    # Split
    from data.preprocessing import FEATURE_COLS, TARGET_COL
    X = df_processed[FEATURE_COLS]
    y = df_processed[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # Train and evaluate
    best_model, best_name, results = train_and_evaluate(X_train, X_test, y_train, y_test, scaler)

    # Re-fit scaler (already done inside train_and_evaluate but re-assign)
    X_all_sc, final_scaler = scale_features(X)
    best_model.fit(X_all_sc, y)  # retrain on full data for production

    # Save
    save_artifacts(best_model, final_scaler, label_encoder, models_dir)
    print("\n✅ Training complete!")
