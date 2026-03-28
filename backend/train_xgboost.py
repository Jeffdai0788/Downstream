"""
TrophicTrace — XGBoost PFAS Contamination Screening Model
Stage 1: Predict water-column PFAS concentration from environmental features.
Based on methodology from Paulson et al. (Science, 2024).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import time
import os


FEATURE_COLS = [
    'upstream_npdes_count', 'upstream_npdes_pfas_count',
    'nearest_pfas_facility_km', 'upstream_discharge_volume_m3',
    'pfas_industry_density', 'afff_site_nearby', 'wwtp_upstream', 'landfill_upstream',
    'mean_annual_flow_m3s', 'low_flow_7q10_m3s', 'stream_order',
    'watershed_area_km2', 'baseflow_index', 'mean_velocity_ms',
    'pct_urban', 'pct_agriculture', 'pct_forest', 'pct_impervious',
    'population_density', 'airport_within_10km', 'fire_training_site',
    'ph', 'temperature_c', 'dissolved_organic_carbon_mgl',
    'total_organic_carbon_mgl', 'conductivity_us_cm',
    'latitude', 'longitude',
]

TARGET_COL = 'water_pfas_ng_l'


def train_model(data_path: str = 'training_data.csv', output_dir: str = '.'):
    """Train XGBoost model with 5-fold cross-validation."""
    print("=" * 60)
    print("TrophicTrace — XGBoost PFAS Contamination Screening Model")
    print("=" * 60)

    # Load data
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} training samples")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Target range: {df[TARGET_COL].min():.1f} – {df[TARGET_COL].max():.1f} ng/L")
    print(f"Target median: {df[TARGET_COL].median():.1f} ng/L")

    X = df[FEATURE_COLS].values
    # Log-transform target for better regression (PFAS is log-normally distributed)
    y = np.log1p(df[TARGET_COL].values)

    # XGBoost parameters — based on Paulson et al. (Science, 2024)
    params = {
        'n_estimators': 250,
        'max_depth': 3,
        'learning_rate': 0.05,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1,
    }

    print(f"\nModel parameters: {json.dumps(params, indent=2)}")

    # 5-fold cross-validation
    print("\n--- 5-Fold Cross-Validation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = {
        'rmse': [], 'mae': [], 'r2': [],
        'within_factor_2': [], 'within_factor_3': [],
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)

        # Metrics in original scale
        y_val_orig = np.expm1(y_val)
        y_pred_orig = np.expm1(y_pred)

        rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
        mae = mean_absolute_error(y_val_orig, y_pred_orig)
        r2 = r2_score(y_val_orig, y_pred_orig)

        # Factor accuracy (standard for environmental models)
        ratio = y_pred_orig / np.maximum(y_val_orig, 0.01)
        within_2 = np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100
        within_3 = np.mean((ratio >= 0.33) & (ratio <= 3.0)) * 100

        cv_results['rmse'].append(rmse)
        cv_results['mae'].append(mae)
        cv_results['r2'].append(r2)
        cv_results['within_factor_2'].append(within_2)
        cv_results['within_factor_3'].append(within_3)

        print(f"  Fold {fold+1}: RMSE={rmse:.1f} | MAE={mae:.1f} | R²={r2:.3f} | "
              f"Within 2×={within_2:.1f}% | Within 3×={within_3:.1f}%")

    print(f"\n  Mean: RMSE={np.mean(cv_results['rmse']):.1f} ± {np.std(cv_results['rmse']):.1f} | "
          f"R²={np.mean(cv_results['r2']):.3f} ± {np.std(cv_results['r2']):.3f} | "
          f"Within 3×={np.mean(cv_results['within_factor_3']):.1f}%")

    # Train final model on full data
    print("\n--- Training Final Model ---")
    start_time = time.time()

    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X, y, verbose=False)

    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.2f}s")

    # Feature importance
    importance = final_model.feature_importances_
    feature_imp = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1], reverse=True)

    print("\n--- Top 10 Feature Importances ---")
    for feat, imp in feature_imp[:10]:
        bar = "█" * int(imp * 100)
        print(f"  {feat:35s} {imp:.4f} {bar}")

    # Save model
    model_path = os.path.join(output_dir, 'xgboost_model.json')
    final_model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    # Save metrics
    metrics = {
        'cv_rmse_mean': round(float(np.mean(cv_results['rmse'])), 2),
        'cv_rmse_std': round(float(np.std(cv_results['rmse'])), 2),
        'cv_r2_mean': round(float(np.mean(cv_results['r2'])), 4),
        'cv_r2_std': round(float(np.std(cv_results['r2'])), 4),
        'cv_within_factor_3_mean': round(float(np.mean(cv_results['within_factor_3'])), 2),
        'train_time_seconds': round(train_time, 2),
        'n_samples': len(df),
        'n_features': len(FEATURE_COLS),
        'n_estimators': params['n_estimators'],
        'feature_importance': [{"feature": f, "importance": round(float(i), 4)}
                                for f, i in feature_imp],
    }

    metrics_path = os.path.join(output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return final_model, metrics


def predict_national(model_path: str = 'xgboost_model.json',
                      data_path: str = 'training_data.csv') -> pd.DataFrame:
    """Run inference on all segments."""
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    df = pd.read_csv(data_path)
    X = df[FEATURE_COLS].values

    start = time.time()
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    elapsed = time.time() - start

    df['predicted_water_pfas_ng_l'] = np.round(y_pred, 2)

    print(f"Inference on {len(df)} segments: {elapsed:.3f}s")
    print(f"Predicted range: {y_pred.min():.1f} – {y_pred.max():.1f} ng/L")

    return df


if __name__ == '__main__':
    model, metrics = train_model()
    print("\n=== Training Complete ===")
    print(f"CV R²: {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
    print(f"Within factor of 3: {metrics['cv_within_factor_3_mean']:.1f}%")
