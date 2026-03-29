#!/usr/bin/env python3
"""
TrophicTrace — Build Real Training Data from Bulk CSVs
======================================================
Builds training_data_real.csv entirely from local data:
  - WQP bulk PFAS measurements (54K records, data_gen/data/wqp_all_pfas.csv)
  - WQP station metadata (data_gen/data/wqp_stations.csv)
  - Pre-fetched hotspot environmental data (new_data/real_data/)
  - API cache from previous NLDI/StreamCat/ECHO queries

NO live API calls needed. Runs in seconds.

Output: backend/training_data_real.csv
"""

import json
import math
import os
import numpy as np
import pandas as pd
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = PROJECT_DIR / "data_gen" / "data"
HOTSPOT_DIR = PROJECT_DIR / "new_data" / "real_data"
CACHE_DIR = BACKEND_DIR / "api_cache"
OUTPUT_CSV = BACKEND_DIR / "training_data_real.csv"

# Target congeners
CONGENERS = ['PFOS', 'PFOA', 'PFNA', 'PFHxS', 'PFDA', 'GenX']

# Surface water type keywords
SW_KEYWORDS = ["River", "Stream", "Lake", "Estuary", "Reservoir", "Creek"]

# Non-detect patterns to exclude
ND_PATTERNS = ["Not Detected", "Below", "Not Detected at Reporting Limit"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_hotspot_data():
    """Load enriched environmental data from pre-fetched hotspot JSON."""
    path = HOTSPOT_DIR / "enriched_environmental_data.json"
    if not path.exists():
        print(f"  Warning: {path} not found")
        return []
    with open(path) as f:
        return json.load(f).get("hotspots", [])


def load_pfas_facilities(hotspots):
    """Extract all facility locations from hotspot data."""
    facilities = []
    for hs in hotspots:
        for fac in hs.get("pfas_facilities", []):
            lat, lng = fac.get("lat"), fac.get("lng")
            if lat and lng:
                is_dup = any(haversine_km(lat, lng, f[0], f[1]) < 0.5
                             for f in facilities)
                if not is_dup:
                    facilities.append((lat, lng))
    return facilities


def load_cached_api_data():
    """Load any previously cached NLDI/StreamCat API responses for enrichment."""
    cache_data = {}  # {(lat_rounded, lon_rounded): {comid, area, urban}}
    if not CACHE_DIR.exists():
        return cache_data

    # Parse cached NLDI position responses (coord -> COMID)
    comid_by_coord = {}
    area_by_comid = {}
    urban_by_comid = {}

    for f in CACHE_DIR.iterdir():
        if not f.name.endswith('.json'):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        fname = f.name

        # NLDI position response → COMID
        if 'comid' in fname and 'position' in fname:
            if isinstance(data, dict) and 'features' in data and data['features']:
                props = data['features'][0].get('properties', {})
                comid = props.get('comid') or props.get('identifier')
                # Extract coords from the URL in the filename
                # This is fragile but works for our cache format
                if comid:
                    comid_by_coord[f.stem] = str(comid)

        # NLDI tot response → watershed area
        if '/tot' in fname or 'tot' in fname:
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        area = item.get('TOT_BASIN_AREA')
                        if area:
                            # Extract COMID from filename
                            area_by_comid[f.stem] = float(area)

        # StreamCat response → urban %
        if 'StreamCat' in fname or 'streamcat' in fname.lower():
            if isinstance(data, dict) and 'items' in data and data['items']:
                item = data['items'][0]
                urban = item.get('pcturbmd2019ws') or item.get('pcturbmd2019cat')
                comid = item.get('comid')
                if urban is not None and comid:
                    urban_by_comid[str(comid)] = float(urban)

    print(f"  Cached COMIDs: {len(comid_by_coord)}")
    print(f"  Cached watershed areas: {len(area_by_comid)}")
    print(f"  Cached urban %: {len(urban_by_comid)}")
    return {'comids': comid_by_coord, 'areas': area_by_comid, 'urban': urban_by_comid}


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


def build():
    print("=" * 60)
    print("TrophicTrace — Build Training Data from Bulk CSVs")
    print("=" * 60)

    # ── 1. Load & process WQP measurements ──────────────────────────────
    print("\n--- Step 1: Load WQP measurements ---")
    wqp = pd.read_csv(DATA_DIR / "wqp_all_pfas.csv",
                       usecols=['MonitoringLocationIdentifier', 'ActivityStartDate',
                                'ResultMeasureValue', 'ResultMeasure/MeasureUnitCode',
                                'ResultDetectionConditionText', '_congener_key'],
                       low_memory=False)
    print(f"  Loaded {len(wqp):,} WQP records")

    # Filter to our 6 congeners
    wqp = wqp[wqp['_congener_key'].isin(CONGENERS)].copy()
    print(f"  {len(wqp):,} records for 6 target congeners")

    # Remove non-detects
    det_cond = wqp['ResultDetectionConditionText'].fillna('')
    nd_mask = det_cond.str.contains('|'.join(ND_PATTERNS), case=False, na=False)
    wqp = wqp[~nd_mask].copy()
    print(f"  {len(wqp):,} after removing non-detects")

    # Parse values and normalize to ng/L
    wqp['value_num'] = to_numeric_safe(wqp['ResultMeasureValue'])
    wqp = wqp.dropna(subset=['value_num'])
    wqp = wqp[wqp['value_num'] > 0].copy()
    wqp['conc_ngl'] = normalize_to_ngl(wqp['value_num'], wqp['ResultMeasure/MeasureUnitCode'])
    wqp = wqp.dropna(subset=['conc_ngl']).copy()
    print(f"  {len(wqp):,} with valid positive concentrations")

    # Parse dates
    wqp['date'] = pd.to_datetime(wqp['ActivityStartDate'], errors='coerce')
    wqp['month'] = wqp['date'].dt.month
    wqp['year'] = wqp['date'].dt.year

    # Aggregate: total PFAS per station per sampling event (sum congeners)
    events = wqp.groupby(['MonitoringLocationIdentifier', 'ActivityStartDate']).agg(
        water_pfas_ng_l=('conc_ngl', 'sum'),
        n_congeners=('conc_ngl', 'count'),
        month=('month', 'first'),
        year=('year', 'first'),
    ).reset_index()
    print(f"  {len(events):,} unique station-date sampling events")

    # ── 2. Load station metadata ──────────────────────────────────────
    print("\n--- Step 2: Load station metadata ---")
    sta = pd.read_csv(DATA_DIR / "wqp_stations.csv",
                       usecols=['MonitoringLocationIdentifier', 'LatitudeMeasure',
                                'LongitudeMeasure', 'HUCEightDigitCode',
                                'DrainageAreaMeasure/MeasureValue',
                                'DrainageAreaMeasure/MeasureUnitCode',
                                'MonitoringLocationTypeName', 'StateCode'],
                       low_memory=False)

    # Filter to surface water
    sw_mask = sta['MonitoringLocationTypeName'].astype(str).apply(
        lambda t: any(kw.lower() in t.lower() for kw in SW_KEYWORDS)
    )
    sta = sta[sw_mask].copy()
    print(f"  {len(sta):,} surface water stations")

    # Parse drainage area
    da_val = pd.to_numeric(sta['DrainageAreaMeasure/MeasureValue'], errors='coerce')
    da_unit = sta['DrainageAreaMeasure/MeasureUnitCode'].astype(str).str.lower()
    da_km2 = da_val.copy()
    da_km2[da_unit.isin(['sq mi', 'square miles', 'mi2'])] = da_val * 2.58999
    da_km2[da_unit.isin(['acres', 'acre'])] = da_val * 0.00404686
    sta['watershed_area_km2'] = da_km2

    # Merge events with station metadata
    merged = events.merge(sta.drop_duplicates('MonitoringLocationIdentifier'),
                          on='MonitoringLocationIdentifier', how='inner')
    merged = merged[merged['LatitudeMeasure'].notna() & merged['LongitudeMeasure'].notna()].copy()
    print(f"  {len(merged):,} events with station coordinates")

    # ── 3. Enrich with hotspot-derived features ────────────────────────
    print("\n--- Step 3: Enrich with environmental features ---")
    hotspots = load_hotspot_data()
    facilities = load_pfas_facilities(hotspots)
    print(f"  {len(hotspots)} hotspots, {len(facilities)} PFAS facilities loaded")

    lats = merged['LatitudeMeasure'].values
    lngs = merged['LongitudeMeasure'].values
    n = len(merged)

    # Facility features: count within 50km + nearest distance
    fac_counts = np.zeros(n, dtype=int)
    fac_dists = np.full(n, 999.0)
    for i in range(n):
        for flat, flng in facilities:
            d = haversine_km(lats[i], lngs[i], flat, flng)
            if d < 50:
                fac_counts[i] += 1
            if d < fac_dists[i]:
                fac_dists[i] = d

    merged['upstream_pfas_facility_count'] = fac_counts
    merged['nearest_pfas_facility_km'] = np.round(fac_dists, 2)

    # Nearest hotspot for fallback features
    def nearest_hotspot_features(lat, lon):
        best = None
        best_d = float('inf')
        for hs in hotspots:
            d = haversine_km(lat, lon, hs['lat'], hs['lng'])
            if d < best_d:
                best_d = d
                best = hs
        return best, best_d

    # Urban % and flow from nearest hotspot (weighted by distance)
    pct_urban = np.full(n, 15.0)
    mean_flow = np.full(n, 50.0)
    for i in range(n):
        hs, d = nearest_hotspot_features(lats[i], lngs[i])
        if hs:
            # Distance-weighted interpolation toward national median
            weight = max(0.0, 1.0 - d / 500.0)  # hotspot influence fades at 500km
            pct_urban[i] = hs.get('pct_urban', 15.0) * weight + 15.0 * (1 - weight)
            mean_flow[i] = hs.get('mean_annual_flow_m3s', 50.0) * weight + 50.0 * (1 - weight)

    merged['pct_urban'] = np.round(pct_urban, 1)

    # Use real watershed area where available, else estimate from flow
    # Fill missing watershed areas with flow-based estimate
    missing_area = merged['watershed_area_km2'].isna()
    merged.loc[missing_area, 'watershed_area_km2'] = np.round(mean_flow[missing_area.values] * 10, 1)

    merged['mean_annual_flow_m3s'] = np.round(mean_flow, 2)

    # ── 4. Build monthly time series (12 months) ──────────────────────
    print("\n--- Step 4: Build monthly aggregation ---")
    # For the time dimension, compute monthly average PFAS across all stations
    monthly_avg = merged.groupby('month')['water_pfas_ng_l'].agg(['mean', 'median', 'count'])
    print("  Monthly median PFAS (ng/L):")
    for m in range(1, 13):
        if m in monthly_avg.index:
            row = monthly_avg.loc[m]
            print(f"    Month {m:2d}: median={row['median']:6.1f}, n={int(row['count'])}")

    # ── 5. Save training data ────────────────────────────────────────
    print("\n--- Step 5: Save training data ---")

    # Use month from the actual sampling event
    merged['month'] = merged['month'].fillna(6).astype(int)

    training = pd.DataFrame({
        'station_id': merged['MonitoringLocationIdentifier'],
        'latitude': np.round(merged['LatitudeMeasure'].values, 6),
        'longitude': np.round(merged['LongitudeMeasure'].values, 6),
        'upstream_pfas_facility_count': merged['upstream_pfas_facility_count'],
        'nearest_pfas_facility_km': merged['nearest_pfas_facility_km'],
        'watershed_area_km2': np.round(merged['watershed_area_km2'].values, 1),
        'pct_urban': merged['pct_urban'],
        'mean_annual_flow_m3s': merged['mean_annual_flow_m3s'],
        'month': merged['month'],
        'water_pfas_ng_l': np.round(merged['water_pfas_ng_l'].values, 2),
        # Metadata (not model features)
        'huc8': merged['HUCEightDigitCode'],
        'sample_date': merged['ActivityStartDate'],
        'year': merged['year'],
        'n_congeners': merged['n_congeners'],
    })

    # Remove extreme outliers (> 99.9th percentile — likely lab errors)
    p999 = training['water_pfas_ng_l'].quantile(0.999)
    n_before = len(training)
    training = training[training['water_pfas_ng_l'] <= p999].copy()
    print(f"  Removed {n_before - len(training)} extreme outliers (> {p999:.0f} ng/L)")

    training.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'=' * 60}")
    print(f"Saved {len(training):,} training samples → {OUTPUT_CSV}")
    print(f"\nFeature summary:")
    for col in ['latitude', 'longitude', 'upstream_pfas_facility_count',
                'nearest_pfas_facility_km', 'watershed_area_km2', 'pct_urban',
                'mean_annual_flow_m3s', 'month']:
        vals = training[col]
        print(f"  {col:35s} min={vals.min():10.2f}  median={vals.median():10.2f}  max={vals.max():10.2f}")

    pfas = training['water_pfas_ng_l']
    print(f"\nTarget (water_pfas_ng_l):")
    print(f"  Range: {pfas.min():.2f} – {pfas.max():.2f} ng/L")
    print(f"  Median: {pfas.median():.2f} ng/L")
    print(f"  Mean: {pfas.mean():.2f} ng/L")
    print(f"  Samples by year: {training.groupby('year').size().to_dict()}")
    print(f"{'=' * 60}")

    # ── 6. Save monthly flow patterns for time slider ──────────────────
    print("\n--- Step 6: Save seasonal patterns ---")
    seasonal = {}
    for hs in hotspots:
        flows = hs.get('monthly_mean_flows_m3s', {})
        if flows:
            annual_mean = hs.get('mean_annual_flow_m3s', 1.0)
            # Seasonal ratio: each month's flow relative to annual mean
            # Higher flow = more dilution = lower PFAS (inverse relationship)
            ratios = {}
            for month_name, flow in flows.items():
                month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                if month_name in month_names:
                    m = month_names.index(month_name) + 1
                    ratios[m] = round(annual_mean / max(flow, 1.0), 3)  # inverse: low flow = high ratio
            seasonal[hs['hotspot_id']] = {
                'name': hs['hotspot_name'],
                'lat': hs['lat'],
                'lng': hs['lng'],
                'monthly_pfas_multiplier': ratios,
                'mean_annual_flow_m3s': annual_mean,
            }

    seasonal_path = BACKEND_DIR / "seasonal_patterns.json"
    with open(seasonal_path, 'w') as f:
        json.dump(seasonal, f, indent=2)
    print(f"  Saved seasonal flow patterns for {len(seasonal)} hotspots → {seasonal_path}")

    return training


if __name__ == "__main__":
    build()
