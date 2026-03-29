#!/usr/bin/env python3
"""
TrophicTrace — Build Monthly PFAS Timeseries from Real WQP Measurements
========================================================================
For each WQP station with PFAS detections:
  1. Average real measurements by calendar month
  2. Linearly interpolate between measured months
  3. Hold CONSTANT before the first measured month and after the last
  4. Measured months always hit their exact average values

Also embeds per-segment timeseries into national_results.json by finding
the nearest station to each segment.

Sources: EPA Water Quality Portal bulk PFAS data (54,246 records, 2003-2025)
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = PROJECT_DIR / "data_gen" / "data"

# Same filtering logic as build_training_data.py
CONGENERS = ['PFOS', 'PFOA', 'PFNA', 'PFHxS', 'PFDA', 'GenX']
SW_KEYWORDS = ["River", "Stream", "Lake", "Estuary", "Reservoir", "Creek"]
ND_PATTERNS = ["Not Detected", "Below", "Not Detected at Reporting Limit"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def to_numeric_safe(series):
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^[<>~≈]+\s*", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def normalize_to_ngl(value, unit):
    unit_lower = unit.astype(str).str.strip().str.lower()
    multiplier = pd.Series(np.nan, index=value.index)
    multiplier[unit_lower.isin(["ng/l"])] = 1.0
    multiplier[unit_lower.isin(["ug/l", "µg/l", "\xb5g/l"])] = 1000.0
    multiplier[unit_lower.isin(["mg/l"])] = 1_000_000.0
    return value * multiplier


def interpolate_monthly(month_avg: dict) -> dict:
    """
    Given {month_int: avg_pfas} for measured months, produce a full 12-month
    timeseries with:
      - Exact values at measured months
      - Linear interpolation between measured months
      - Constant extrapolation before first / after last measured month
    """
    if not month_avg:
        return {str(m): 0.0 for m in range(1, 13)}

    measured_months = sorted(month_avg.keys())

    if len(measured_months) == 1:
        # Single measurement — constant across all months
        val = month_avg[measured_months[0]]
        return {str(m): round(val, 2) for m in range(1, 13)}

    first_month = measured_months[0]
    last_month = measured_months[-1]

    # Build arrays for np.interp: only the measured months
    xp = np.array(measured_months, dtype=float)
    fp = np.array([month_avg[m] for m in measured_months], dtype=float)

    result = {}
    for m in range(1, 13):
        if m < first_month:
            # Constant extrapolation: hold at first measured value
            result[str(m)] = round(float(fp[0]), 2)
        elif m > last_month:
            # Constant extrapolation: hold at last measured value
            result[str(m)] = round(float(fp[-1]), 2)
        else:
            # Linear interpolation (np.interp handles exact matches correctly)
            val = np.interp(float(m), xp, fp)
            result[str(m)] = round(float(val), 2)

    return result


def build_timeseries():
    print("=" * 60)
    print("TrophicTrace — Build Monthly PFAS Timeseries")
    print("=" * 60)

    # ── 1. Load & process WQP measurements (same as build_training_data.py) ──
    print("\n--- Step 1: Load WQP measurements ---")
    wqp = pd.read_csv(DATA_DIR / "wqp_all_pfas.csv",
                       usecols=['MonitoringLocationIdentifier', 'ActivityStartDate',
                                'ResultMeasureValue', 'ResultMeasure/MeasureUnitCode',
                                'ResultDetectionConditionText', '_congener_key'],
                       low_memory=False)
    print(f"  Loaded {len(wqp):,} WQP records")

    wqp = wqp[wqp['_congener_key'].isin(CONGENERS)].copy()
    det_cond = wqp['ResultDetectionConditionText'].fillna('')
    nd_mask = det_cond.str.contains('|'.join(ND_PATTERNS), case=False, na=False)
    wqp = wqp[~nd_mask].copy()

    wqp['value_num'] = to_numeric_safe(wqp['ResultMeasureValue'])
    wqp = wqp.dropna(subset=['value_num'])
    wqp = wqp[wqp['value_num'] > 0].copy()
    wqp['conc_ngl'] = normalize_to_ngl(wqp['value_num'], wqp['ResultMeasure/MeasureUnitCode'])
    wqp = wqp.dropna(subset=['conc_ngl']).copy()

    wqp['date'] = pd.to_datetime(wqp['ActivityStartDate'], errors='coerce')
    wqp['month'] = wqp['date'].dt.month
    print(f"  {len(wqp):,} valid PFAS detections")

    # ── 2. Sum congeners per station per sampling event ──
    events = wqp.groupby(['MonitoringLocationIdentifier', 'ActivityStartDate']).agg(
        total_pfas_ng_l=('conc_ngl', 'sum'),
        month=('month', 'first'),
    ).reset_index()
    print(f"  {len(events):,} sampling events")

    # ── 3. Load station metadata for coordinates ──
    print("\n--- Step 2: Load station coordinates ---")
    sta = pd.read_csv(DATA_DIR / "wqp_stations.csv",
                       usecols=['MonitoringLocationIdentifier', 'LatitudeMeasure',
                                'LongitudeMeasure', 'HUCEightDigitCode',
                                'MonitoringLocationTypeName'],
                       low_memory=False)

    sw_mask = sta['MonitoringLocationTypeName'].astype(str).apply(
        lambda t: any(kw.lower() in t.lower() for kw in SW_KEYWORDS)
    )
    sta = sta[sw_mask].copy()

    events = events.merge(sta.drop_duplicates('MonitoringLocationIdentifier'),
                          on='MonitoringLocationIdentifier', how='inner')
    events = events.dropna(subset=['LatitudeMeasure', 'LongitudeMeasure'])
    print(f"  {events['MonitoringLocationIdentifier'].nunique()} stations with coordinates")

    # ── 4. Build monthly averages per station ──
    print("\n--- Step 3: Build per-station monthly timeseries ---")
    station_monthly = events.groupby(['MonitoringLocationIdentifier', 'month']).agg(
        avg_pfas=('total_pfas_ng_l', 'mean'),
        n_events=('total_pfas_ng_l', 'count'),
    ).reset_index()

    # Get station-level metadata
    station_meta = events.groupby('MonitoringLocationIdentifier').agg(
        lat=('LatitudeMeasure', 'first'),
        lng=('LongitudeMeasure', 'first'),
        huc8=('HUCEightDigitCode', 'first'),
        n_total_events=('total_pfas_ng_l', 'count'),
        latest_date=('ActivityStartDate', 'max'),
    ).reset_index()

    # Build timeseries for each station
    stations_out = []
    n_interpolated = 0
    n_constant = 0

    for _, meta in station_meta.iterrows():
        sid = meta['MonitoringLocationIdentifier']
        sm = station_monthly[station_monthly['MonitoringLocationIdentifier'] == sid]

        month_avg = {int(row['month']): float(row['avg_pfas']) for _, row in sm.iterrows()}
        monthly_ts = interpolate_monthly(month_avg)

        n_measured = len(month_avg)
        if n_measured > 1:
            n_interpolated += 1
        else:
            n_constant += 1

        # Collect real measurement dates/values for provenance
        sta_events = events[events['MonitoringLocationIdentifier'] == sid]
        real_measurements = [
            {"date": str(row['ActivityStartDate']), "pfas_ng_l": round(float(row['total_pfas_ng_l']), 2)}
            for _, row in sta_events.iterrows()
        ]

        latest_year = None
        try:
            latest_year = int(str(meta['latest_date'])[:4])
        except (ValueError, TypeError):
            pass

        stations_out.append({
            "station_id": sid,
            "lat": round(float(meta['lat']), 6),
            "lng": round(float(meta['lng']), 6),
            "huc8": str(meta['huc8']) if pd.notna(meta['huc8']) else None,
            "n_real_measurements": int(meta['n_total_events']),
            "n_measured_months": n_measured,
            "latest_year": latest_year,
            "monthly_pfas_ng_l": monthly_ts,
            "real_measurements": sorted(real_measurements, key=lambda x: x['date']),
        })

    print(f"  {len(stations_out)} stations total")
    print(f"  {n_interpolated} with interpolated timeseries (>1 measured month)")
    print(f"  {n_constant} with constant timeseries (1 measured month)")

    # ── 5. Save standalone timeseries JSON ──
    output = {
        "metadata": {
            "method": "Monthly average of real WQP measurements per station. "
                      "Linear interpolation between measured months. "
                      "Constant extrapolation before first / after last measured month.",
            "n_stations": len(stations_out),
            "n_interpolated": n_interpolated,
            "n_constant": n_constant,
            "source": "EPA Water Quality Portal bulk PFAS data (54,246 records, 2003-2025)",
        },
        "stations": stations_out,
    }

    ts_path = BACKEND_DIR / "monthly_timeseries.json"
    with open(ts_path, 'w') as f:
        json.dump(output, f)
    size_kb = ts_path.stat().st_size / 1024
    print(f"\n  Saved {ts_path.name} ({size_kb:.0f} KB)")

    return output


def embed_timeseries_in_results(ts_data: dict, results_path: str = None):
    """
    Attach nearest-station monthly timeseries to each segment in national_results.json.
    """
    if results_path is None:
        results_path = BACKEND_DIR / "national_results.json"

    print(f"\n--- Step 4: Embed timeseries in {Path(results_path).name} ---")

    with open(results_path) as f:
        results = json.load(f)

    stations = ts_data['stations']
    # Build lookup arrays for fast nearest-station search
    sta_lats = np.array([s['lat'] for s in stations])
    sta_lngs = np.array([s['lng'] for s in stations])

    n_embedded = 0
    for seg in results['segments']:
        seg_lat, seg_lng = seg['lat'], seg['lng']

        # Find nearest station (Euclidean approximation for speed, fine at this scale)
        dists = np.sqrt((sta_lats - seg_lat)**2 + (sta_lngs - seg_lng)**2)
        nearest_idx = int(np.argmin(dists))
        nearest_sta = stations[nearest_idx]
        nearest_km = haversine_km(seg_lat, seg_lng, nearest_sta['lat'], nearest_sta['lng'])

        # Attach timeseries with risk levels
        monthly = nearest_sta['monthly_pfas_ng_l']
        monthly_with_risk = {}
        for m_str, val in monthly.items():
            if val > 40:
                risk = "high"
            elif val > 8:
                risk = "medium"
            else:
                risk = "low"
            monthly_with_risk[m_str] = {"water_pfas_ng_l": val, "risk_level": risk}

        seg['monthly_timeseries'] = monthly_with_risk
        seg['nearest_station_km'] = round(nearest_km, 1)
        seg['timeseries_station_id'] = nearest_sta['station_id']
        seg['timeseries_method'] = (
            "interpolated" if nearest_sta['n_measured_months'] > 1 else "constant"
        )
        seg['n_real_measurements'] = nearest_sta['n_real_measurements']
        n_embedded += 1

    with open(results_path, 'w') as f:
        json.dump(results, f)

    size_mb = Path(results_path).stat().st_size / (1024 * 1024)
    print(f"  Embedded timeseries in {n_embedded} segments → {Path(results_path).name} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    ts_data = build_timeseries()
    embed_timeseries_in_results(ts_data)
    print("\n=== Timeseries Build Complete ===")
