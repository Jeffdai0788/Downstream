"""
TrophicTrace — Gradient Boosted PFAS Contamination Screening Model (v2)
Stage 1: Predict water-column PFAS concentration from 8 real features.
Trained on 2,131 real EPA WQP measurements from bulk federal CSV data.
Based on methodology from Paulson et al. (Science, 2024).
Uses sklearn GradientBoostingRegressor (no OpenMP dependency).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import time
import os


# 8 pruned features — all from real federal data sources
FEATURE_COLS = [
    'latitude',                       # WQP station coordinates
    'longitude',                      # WQP station coordinates
    'upstream_pfas_facility_count',   # EPA TRI/ECHO real facilities within 50km
    'nearest_pfas_facility_km',       # Haversine to nearest real PFAS facility
    'watershed_area_km2',             # WQP drainage area / NHDPlus
    'pct_urban',                      # StreamCat NLCD 2019 urban land cover %
    'mean_annual_flow_m3s',           # USGS NWIS / hotspot flow data
    'month',                          # WQP sample_date — seasonal signal
]

TARGET_COL = 'water_pfas_ng_l'


def train_model(data_path: str = 'training_data_real.csv', output_dir: str = '.'):
    """Train GBR on real WQP PFAS measurements with 8 real features."""
    print("=" * 60)
    print("TrophicTrace — GBR PFAS Screening (v2, Real Data)")
    print("=" * 60)

    data_path_full = os.path.join(output_dir, data_path) if not os.path.isabs(data_path) else data_path
    if not os.path.exists(data_path_full):
        data_path_full = data_path

    df = pd.read_csv(data_path_full)
    print(f"\nLoaded {len(df)} training samples from {data_path_full}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Target range: {df[TARGET_COL].min():.2f} – {df[TARGET_COL].max():.2f} ng/L")
    print(f"Target median: {df[TARGET_COL].median():.2f} ng/L")

    X = df[FEATURE_COLS].values
    y = np.log1p(df[TARGET_COL].values)

    params = {
        'n_estimators': 150,
        'max_depth': 3,
        'learning_rate': 0.05,
        'min_samples_leaf': 10,
        'subsample': 0.8,
        'max_features': 0.8,
        'random_state': 42,
    }

    print(f"\nModel: sklearn GradientBoostingRegressor")
    print(f"Parameters: {json.dumps(params, indent=2)}")

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

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        y_val_orig = np.expm1(y_val)
        y_pred_orig = np.expm1(y_pred)

        rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
        mae = mean_absolute_error(y_val_orig, y_pred_orig)
        r2 = r2_score(y_val_orig, y_pred_orig)

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
    final_model = GradientBoostingRegressor(**params)
    final_model.fit(X, y)
    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.2f}s")

    # Feature importance
    importance = final_model.feature_importances_
    feature_imp = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1], reverse=True)

    print("\n--- Feature Importances ---")
    for feat, imp in feature_imp:
        bar = "█" * int(imp * 100)
        print(f"  {feat:35s} {imp:.4f} {bar}")

    # Save model
    model_path = os.path.join(output_dir, 'gbr_model.joblib')
    joblib.dump(final_model, model_path)
    print(f"\nModel saved to {model_path}")

    # Save metrics
    metrics = {
        'model_version': '2.0_real_data',
        'model_type': 'sklearn.ensemble.GradientBoostingRegressor',
        'training_data': {
            'n_samples': len(df),
            'data_source': 'EPA WQP bulk PFAS measurements (54K records, 2003-2025)',
            'n_features': len(FEATURE_COLS),
            'features': FEATURE_COLS,
            'feature_sources': {
                'latitude': 'WQP station coordinates',
                'longitude': 'WQP station coordinates',
                'upstream_pfas_facility_count': 'EPA TRI/ECHO real facilities (38 geocoded, 50km radius)',
                'nearest_pfas_facility_km': 'Haversine distance to nearest real PFAS facility',
                'watershed_area_km2': 'WQP drainage area + NHDPlus estimates',
                'pct_urban': 'EPA StreamCat NLCD 2019 (via hotspot interpolation)',
                'mean_annual_flow_m3s': 'USGS NWIS gauge data (via hotspot interpolation)',
                'month': 'WQP sample_date (seasonal signal)',
            },
        },
        'cv_rmse_mean': round(float(np.mean(cv_results['rmse'])), 2),
        'cv_rmse_std': round(float(np.std(cv_results['rmse'])), 2),
        'cv_mae_mean': round(float(np.mean(cv_results['mae'])), 2),
        'cv_r2_mean': round(float(np.mean(cv_results['r2'])), 4),
        'cv_r2_std': round(float(np.std(cv_results['r2'])), 4),
        'cv_within_factor_2_mean': round(float(np.mean(cv_results['within_factor_2'])), 1),
        'cv_within_factor_3_mean': round(float(np.mean(cv_results['within_factor_3'])), 1),
        'train_time_seconds': round(train_time, 2),
        'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'feature_importance': [{"feature": f, "importance": round(float(i), 4)}
                                for f, i in feature_imp],
    }

    metrics_path = os.path.join(output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return final_model, metrics


def load_model(model_path: str = 'gbr_model.joblib'):
    """Load trained model."""
    return joblib.load(model_path)


def predict_national(model_path: str = 'gbr_model.joblib',
                      data_path: str = 'training_data_real.csv') -> pd.DataFrame:
    """Run inference on all segments."""
    model = load_model(model_path)
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
