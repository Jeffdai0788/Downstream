#!/usr/bin/env python3
"""
TrophicTrace — Real Data Enrichment Pipeline
=============================================
Enriches 905 WQP monitoring stations with real environmental features
from federal APIs, producing training data with real PFAS targets.

Data sources:
  - NLDI (api.water.usgs.gov/nldi): coord → COMID → watershed area
  - StreamCat (api.epa.gov/StreamCat): COMID → urban land cover %
  - EPA ECHO (echodata.epa.gov): lat/lon → nearby PFAS facilities
  - USGS Water Services (waterservices.usgs.gov): historical flow data
  - Hotspot enrichment data (new_data/real_data/): pre-fetched facility/flow/species data

Also uses pre-fetched hotspot data from new_data/real_data/enriched_environmental_data.json
to supplement API results and provide fallback values.

Output: backend/training_data_real.csv (905 rows × 8 features + real PFAS target)
"""

import json
import math
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
VALIDATION_CSV = BACKEND_DIR / "validation_real_data.csv"
ENRICHED_HOTSPOT = PROJECT_DIR / "new_data" / "real_data" / "enriched_environmental_data.json"
OUTPUT_CSV = BACKEND_DIR / "training_data_real.csv"
CACHE_DIR = BACKEND_DIR / "api_cache"
CACHE_DIR.mkdir(exist_ok=True)

TIMEOUT = 30
RATE_LIMIT_SEC = 0.25  # 4 req/sec max

# ── Feature columns for the pruned 8-feature model ───────────────────────────
FEATURE_COLS = [
    'latitude',
    'longitude',
    'upstream_pfas_facility_count',
    'nearest_pfas_facility_km',
    'watershed_area_km2',
    'pct_urban',
    'mean_annual_flow_m3s',
    'month',
]
TARGET_COL = 'water_pfas_ng_l'


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_json(url, label="", retries=2):
    """Fetch JSON from URL with timeout, retries, and caching."""
    # Check cache first
    cache_key = urllib.parse.quote(url, safe='')[:200]
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    req = urllib.request.Request(url, headers={
        "User-Agent": "TrophicTrace/2.0 (YHack 2026 Research, contact: andrew@yhack.org)"
    })

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                # Cache the result
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
                return data
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception:
            if attempt < retries:
                time.sleep(1)
                continue
            return None
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Pre-load hotspot data for fallback/interpolation ─────────────────────────
def load_hotspot_data():
    """Load enriched hotspot data for spatial interpolation fallback."""
    if not ENRICHED_HOTSPOT.exists():
        print(f"  Warning: {ENRICHED_HOTSPOT} not found — no hotspot fallback")
        return []
    with open(ENRICHED_HOTSPOT) as f:
        data = json.load(f)
    return data.get("hotspots", [])


def nearest_hotspot(lat, lon, hotspots):
    """Find nearest hotspot and return it with distance."""
    best_dist = float('inf')
    best = None
    for hs in hotspots:
        d = haversine_km(lat, lon, hs["lat"], hs["lng"])
        if d < best_dist:
            best_dist = d
            best = hs
    return best, best_dist


# ═══════════════════════════════════════════════════════════════════════════════
# API Enrichment Functions
# ═══════════════════════════════════════════════════════════════════════════════

def get_comid(lat, lon):
    """Get NHDPlus COMID for a coordinate via NLDI."""
    url = (f"https://api.water.usgs.gov/nldi/linked-data/comid/position"
           f"?coords=POINT({lon}%20{lat})")
    data = fetch_json(url)
    if data and "features" in data and len(data["features"]) > 0:
        props = data["features"][0].get("properties", {})
        return props.get("comid") or props.get("identifier")
    return None


def get_watershed_area(comid):
    """Get total watershed area from NLDI basin characteristics."""
    url = f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}/tot"
    data = fetch_json(url)
    if data and isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # Look for basin area characteristic
                val = item.get("TOT_BASIN_AREA")
                if val is not None:
                    return float(val)
    # Try as dict
    if data and isinstance(data, dict):
        val = data.get("TOT_BASIN_AREA")
        if val is not None:
            return float(val)
        # Check nested
        chars = data.get("characteristics") or data.get("items") or []
        if isinstance(chars, list):
            for c in chars:
                if isinstance(c, dict) and c.get("characteristic_id") == "TOT_BASIN_AREA":
                    return float(c.get("characteristic_value", 0))
    return None


def get_urban_pct(comid):
    """Get urban land cover % from EPA StreamCat."""
    url = (f"https://api.epa.gov/StreamCat/streams/metrics"
           f"?name=PctUrbMd2019&comid={comid}")
    data = fetch_json(url)
    if data and "items" in data and len(data["items"]) > 0:
        item = data["items"][0]
        # Prefer watershed-level, fall back to catchment-level
        return item.get("pcturbmd2019ws") or item.get("pcturbmd2019cat")
    return None


def load_pfas_facilities():
    """
    Load all known PFAS facilities from pre-fetched hotspot data.
    These are real geocoded facilities from EPA TRI, ECHO, and DOD PFAS Inventory.
    """
    facilities = []
    if ENRICHED_HOTSPOT.exists():
        with open(ENRICHED_HOTSPOT) as f:
            data = json.load(f)
        for hs in data.get("hotspots", []):
            for fac in hs.get("pfas_facilities", []):
                lat = fac.get("lat")
                lng = fac.get("lng")
                if lat and lng:
                    facilities.append({"lat": lat, "lng": lng, "name": fac.get("name", "")})
    # Also check standalone facilities file
    fac_path = PROJECT_DIR / "new_data" / "real_data" / "pfas_facilities_real.json"
    if fac_path.exists():
        with open(fac_path) as f:
            fac_data = json.load(f)
        for hs_facs in fac_data.get("hotspot_facilities", fac_data.get("hotspots", [])):
            for fac in hs_facs.get("nearby_facilities", hs_facs.get("pfas_facilities", [])):
                lat = fac.get("lat")
                lng = fac.get("lng")
                if lat and lng:
                    # Deduplicate by proximity
                    is_dup = any(haversine_km(lat, lng, f["lat"], f["lng"]) < 0.5 for f in facilities)
                    if not is_dup:
                        facilities.append({"lat": lat, "lng": lng, "name": fac.get("name", "")})
    return facilities


def get_facility_features(lat, lon, all_facilities, radius_km=50):
    """Compute facility count and nearest distance from pre-loaded facility list."""
    count = 0
    min_dist = 999.0
    for fac in all_facilities:
        d = haversine_km(lat, lon, fac["lat"], fac["lng"])
        if d < radius_km:
            count += 1
        if d < min_dist:
            min_dist = d
    return count, round(min_dist, 2)


def get_echo_facility_count(lat, lon, radius_miles=30):
    """Get PFAS-relevant facility count from EPA ECHO (count only, no coords).

    ECHO's QID endpoint doesn't reliably return coordinates, so we only
    use this for the count. Distance comes from pre-loaded facility data.
    """
    sic_codes = "2821,2869,4952,9711"
    url = (f"https://echodata.epa.gov/echo/echo_rest_services.get_facilities"
           f"?output=JSON&p_lat={lat}&p_long={lon}&p_radius={radius_miles}"
           f"&p_sic={sic_codes}")
    data = fetch_json(url)
    if data and "Results" in data:
        return int(data["Results"].get("QueryRows", 0))
    return 0


def get_mean_flow_nwis(lat, lon):
    """
    Find nearest NWIS streamgage and get mean annual flow.
    Uses NLDI to find upstream gages, then queries daily values.
    Returns flow in m3/s or None.
    """
    # First try to find a nearby NWIS site via NLDI
    comid = get_comid(lat, lon)
    if not comid:
        return None

    # Search for NWIS sites on this reach
    url = (f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}"
           f"/navigate/UT/nwissite?distance=50")
    data = fetch_json(url)

    site_id = None
    if data and "features" in data and len(data["features"]) > 0:
        # Take the first (nearest) NWIS site
        props = data["features"][0].get("properties", {})
        identifier = props.get("identifier", "")
        if identifier.startswith("USGS-"):
            site_id = identifier.replace("USGS-", "")

    if not site_id:
        return None

    # Get annual mean flow (parameter 00060 = streamflow in cfs)
    url = (f"https://waterservices.usgs.gov/nwis/dv/"
           f"?format=json&sites={site_id}&parameterCd=00060"
           f"&startDT=2023-01-01&endDT=2024-01-01&statCd=00003")
    data = fetch_json(url)

    if not data or "value" not in data:
        return None

    try:
        ts = data["value"]["timeSeries"]
        if not ts:
            return None
        values = ts[0]["values"][0]["value"]
        flows_cfs = [float(v["value"]) for v in values
                     if float(v["value"]) >= 0]
        if not flows_cfs:
            return None
        mean_cfs = np.mean(flows_cfs)
        # Convert cfs to m3/s (1 cfs = 0.028317 m3/s)
        return round(mean_cfs * 0.028317, 2)
    except (KeyError, IndexError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main Enrichment Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_stations():
    """Enrich 905 WQP stations with 8 real features."""
    print("=" * 70)
    print("TrophicTrace — Real Data Enrichment Pipeline")
    print("=" * 70)

    # Load WQP stations
    df = pd.read_csv(VALIDATION_CSV)
    wqp = df[
        (df['data_source'] == 'WQP') &
        df['latitude'].notna() &
        df['longitude'].notna() &
        df['water_pfas_ng_l'].notna() &
        (df['water_pfas_ng_l'] > 0)
    ].copy().reset_index(drop=True)
    print(f"\nLoaded {len(wqp)} WQP stations with real PFAS measurements")

    # Load hotspot data for fallback
    hotspots = load_hotspot_data()
    print(f"Loaded {len(hotspots)} hotspots for spatial interpolation fallback")

    # Extract month from sample_date
    wqp['month'] = pd.to_datetime(wqp['sample_date'], errors='coerce').dt.month
    wqp['month'] = wqp['month'].fillna(6).astype(int)  # default to June if missing

    # Initialize feature columns
    wqp['upstream_pfas_facility_count'] = np.nan
    wqp['nearest_pfas_facility_km'] = np.nan
    wqp['watershed_area_km2_api'] = np.nan
    wqp['pct_urban'] = np.nan
    wqp['mean_annual_flow_m3s'] = np.nan
    wqp['comid'] = None

    n = len(wqp)
    start_time = time.time()

    # ── Phase 1: NLDI COMID + Watershed Area + StreamCat Urban % ──
    # These are fast APIs — do all stations
    print(f"\n--- Phase 1: NLDI + StreamCat ({n} stations) ---")
    for i in range(n):
        lat = wqp.at[i, 'latitude']
        lon = wqp.at[i, 'longitude']

        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = i / max(elapsed, 1)
            print(f"  Station {i}/{n} ({rate:.1f}/sec)...")

        # Get COMID
        comid = get_comid(lat, lon)
        if comid:
            wqp.at[i, 'comid'] = str(comid)

            # Watershed area from NLDI
            area = get_watershed_area(comid)
            if area:
                wqp.at[i, 'watershed_area_km2_api'] = area

            # Urban % from StreamCat
            urban = get_urban_pct(comid)
            if urban is not None:
                wqp.at[i, 'pct_urban'] = urban

            time.sleep(RATE_LIMIT_SEC)

    comid_hit = wqp['comid'].notna().sum()
    area_hit = wqp['watershed_area_km2_api'].notna().sum()
    urban_hit = wqp['pct_urban'].notna().sum()
    print(f"\n  COMID hits: {comid_hit}/{n}")
    print(f"  Watershed area hits: {area_hit}/{n}")
    print(f"  Urban % hits: {urban_hit}/{n}")

    # ── Phase 2: PFAS Facility Features ──
    # Use pre-loaded real facility data (38 geocoded facilities from EPA TRI/ECHO/DOD)
    # Supplemented with live ECHO count queries for stations far from hotspots
    print(f"\n--- Phase 2: PFAS Facility Features ---")
    all_facilities = load_pfas_facilities()
    print(f"  Loaded {len(all_facilities)} real PFAS facilities from hotspot data")

    echo_cache = {}
    for i in range(n):
        lat = wqp.at[i, 'latitude']
        lon = wqp.at[i, 'longitude']

        # Get count + distance from pre-loaded facilities
        count_local, dist_local = get_facility_features(lat, lon, all_facilities)

        # Also query ECHO for broader count (catches facilities not in hotspot data)
        cache_key = (round(lat, 1), round(lon, 1))
        if cache_key not in echo_cache:
            echo_count = get_echo_facility_count(lat, lon)
            echo_cache[cache_key] = echo_count
            time.sleep(RATE_LIMIT_SEC)
        else:
            echo_count = echo_cache[cache_key]

        # Use max of local + ECHO counts, distance from local (has real coords)
        wqp.at[i, 'upstream_pfas_facility_count'] = max(count_local, echo_count)
        wqp.at[i, 'nearest_pfas_facility_km'] = dist_local

        if i % 100 == 0:
            print(f"  Station {i}/{n}...")

    echo_hit = (wqp['upstream_pfas_facility_count'] > 0).sum()
    print(f"  Stations near PFAS facilities: {echo_hit}/{n}")

    # ── Phase 3: USGS Flow Data ──
    # Most expensive — sample a subset and interpolate
    print(f"\n--- Phase 3: USGS Flow Data (sampling subset) ---")
    # Sample up to 200 stations for flow queries (the rest get interpolated)
    flow_sample_idx = np.random.RandomState(42).choice(n, min(200, n), replace=False)
    flow_count = 0
    for idx in flow_sample_idx:
        lat = wqp.at[idx, 'latitude']
        lon = wqp.at[idx, 'longitude']
        flow = get_mean_flow_nwis(lat, lon)
        if flow is not None:
            wqp.at[idx, 'mean_annual_flow_m3s'] = flow
            flow_count += 1
        time.sleep(RATE_LIMIT_SEC)

        if flow_count % 20 == 0 and flow_count > 0:
            print(f"  Got {flow_count} flow values from {len(flow_sample_idx)} queries...")

    print(f"  Flow data hits: {flow_count}/{len(flow_sample_idx)} sampled")

    # ── Phase 4: Fill missing values ──
    print(f"\n--- Phase 4: Fill missing values ---")

    # Use existing drainage_area from WQP where available
    existing_area = wqp['drainage_area_km2'].notna()
    api_area_missing = wqp['watershed_area_km2_api'].isna()
    wqp.loc[existing_area & api_area_missing, 'watershed_area_km2_api'] = \
        wqp.loc[existing_area & api_area_missing, 'drainage_area_km2']

    # Fill remaining watershed_area from nearest hotspot
    for i in range(n):
        if pd.isna(wqp.at[i, 'watershed_area_km2_api']) and hotspots:
            hs, dist = nearest_hotspot(wqp.at[i, 'latitude'], wqp.at[i, 'longitude'], hotspots)
            if hs:
                wqp.at[i, 'watershed_area_km2_api'] = hs.get('watershed_area_km2', 500.0)
    # Final fallback: median
    median_area = wqp['watershed_area_km2_api'].median()
    wqp['watershed_area_km2_api'] = wqp['watershed_area_km2_api'].fillna(
        median_area if not pd.isna(median_area) else 500.0)

    # Fill pct_urban from nearest hotspot, then median
    for i in range(n):
        if pd.isna(wqp.at[i, 'pct_urban']) and hotspots:
            hs, dist = nearest_hotspot(wqp.at[i, 'latitude'], wqp.at[i, 'longitude'], hotspots)
            if hs:
                wqp.at[i, 'pct_urban'] = hs.get('pct_urban', 15.0)
    median_urban = wqp['pct_urban'].median()
    wqp['pct_urban'] = wqp['pct_urban'].fillna(
        median_urban if not pd.isna(median_urban) else 15.0)

    # Fill flow from nearest hotspot monthly data or watershed-area regression
    for i in range(n):
        if pd.isna(wqp.at[i, 'mean_annual_flow_m3s']) and hotspots:
            hs, dist = nearest_hotspot(wqp.at[i, 'latitude'], wqp.at[i, 'longitude'], hotspots)
            if hs:
                wqp.at[i, 'mean_annual_flow_m3s'] = hs.get('mean_annual_flow_m3s', 50.0)
    # Remaining: estimate from watershed area (rough regression: flow ≈ area * 0.01)
    still_missing = wqp['mean_annual_flow_m3s'].isna()
    wqp.loc[still_missing, 'mean_annual_flow_m3s'] = \
        wqp.loc[still_missing, 'watershed_area_km2_api'] * 0.01

    # Fill facility counts: 0 is a valid value (no facilities nearby)
    wqp['upstream_pfas_facility_count'] = wqp['upstream_pfas_facility_count'].fillna(0)
    wqp['nearest_pfas_facility_km'] = wqp['nearest_pfas_facility_km'].fillna(999.0)

    # ── Phase 5: Build final training DataFrame ──
    print(f"\n--- Phase 5: Build training data ---")
    training_df = pd.DataFrame({
        'station_id': wqp['station_id'],
        'latitude': wqp['latitude'],
        'longitude': wqp['longitude'],
        'upstream_pfas_facility_count': wqp['upstream_pfas_facility_count'].astype(int),
        'nearest_pfas_facility_km': np.round(wqp['nearest_pfas_facility_km'], 2),
        'watershed_area_km2': np.round(wqp['watershed_area_km2_api'], 1),
        'pct_urban': np.round(wqp['pct_urban'], 1),
        'mean_annual_flow_m3s': np.round(wqp['mean_annual_flow_m3s'], 2),
        'month': wqp['month'],
        'water_pfas_ng_l': np.round(wqp['water_pfas_ng_l'], 2),
        # Metadata (not model features)
        'huc8': wqp['huc8'],
        'comid': wqp['comid'],
        'sample_date': wqp['sample_date'],
    })

    # Save
    training_df.to_csv(OUTPUT_CSV, index=False)
    elapsed_total = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Saved {len(training_df)} enriched stations → {OUTPUT_CSV}")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"\nFeature coverage:")
    for col in FEATURE_COLS:
        non_null = training_df[col].notna().sum()
        print(f"  {col:35s} {non_null}/{len(training_df)} ({non_null/len(training_df)*100:.0f}%)")
    print(f"\nTarget (water_pfas_ng_l):")
    print(f"  Range: {training_df['water_pfas_ng_l'].min():.2f} – {training_df['water_pfas_ng_l'].max():.2f} ng/L")
    print(f"  Median: {training_df['water_pfas_ng_l'].median():.2f} ng/L")
    print(f"{'=' * 70}")

    return training_df


if __name__ == "__main__":
    enrich_stations()
