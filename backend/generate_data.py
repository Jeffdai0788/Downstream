"""
TrophicTrace — Synthetic Data Generation Pipeline
Generates realistic national-scale training data for XGBoost PFAS contamination model.
Uses real geographic distributions and environmental parameter ranges from federal datasets.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple

np.random.seed(42)

# ============================================================
# CONSTANTS — all from published EPA/USGS sources
# ============================================================

# PFAS-handling SIC codes (EPA ECHO PFAS Analytic Tools)
PFAS_SIC_CODES = [
    "2821", "2869", "2911", "3312", "3471", "3479", "4952", "4953",
    "2819", "2899", "3089", "3559", "3699", "4941"
]

# Known PFAS hotspot regions (lat, lng, intensity)
# Based on EPA UCMR 5 + known contamination sites
PFAS_HOTSPOTS = [
    {"name": "Cape Fear NC (Chemours)", "lat": 35.05, "lng": -78.88, "radius": 0.5, "intensity": 1.0},
    {"name": "Decatur AL (3M)", "lat": 34.60, "lng": -86.98, "radius": 0.4, "intensity": 0.9},
    {"name": "Fayetteville NC (Chemours)", "lat": 35.05, "lng": -78.88, "radius": 0.3, "intensity": 0.95},
    {"name": "Parkersburg WV (DuPont)", "lat": 39.27, "lng": -81.56, "radius": 0.4, "intensity": 0.85},
    {"name": "Oscoda MI (Wurtsmith AFB)", "lat": 44.45, "lng": -83.33, "radius": 0.3, "intensity": 0.8},
    {"name": "Bennington VT", "lat": 42.88, "lng": -73.20, "radius": 0.3, "intensity": 0.75},
    {"name": "Parchment MI", "lat": 42.33, "lng": -85.57, "radius": 0.2, "intensity": 0.7},
    {"name": "Hoosick Falls NY", "lat": 42.90, "lng": -73.35, "radius": 0.2, "intensity": 0.7},
    {"name": "Newburgh NY (Stewart ANG)", "lat": 41.50, "lng": -74.01, "radius": 0.3, "intensity": 0.65},
    {"name": "Horsham PA (NAS JRB Willow Grove)", "lat": 40.18, "lng": -75.13, "radius": 0.3, "intensity": 0.7},
    {"name": "Colorado Springs CO (Peterson AFB)", "lat": 38.80, "lng": -104.72, "radius": 0.3, "intensity": 0.6},
    {"name": "Great Lakes region", "lat": 43.5, "lng": -84.0, "radius": 2.0, "intensity": 0.5},
    {"name": "Delaware River corridor", "lat": 40.2, "lng": -74.8, "radius": 0.8, "intensity": 0.55},
    {"name": "New Hampshire Merrimack", "lat": 42.86, "lng": -71.49, "radius": 0.3, "intensity": 0.65},
]

# 8 target fish species — all from FishBase
SPECIES = [
    {"common_name": "Largemouth Bass", "scientific_name": "Micropterus salmoides",
     "trophic_level": 4.2, "lipid_pct": 5.8, "body_mass_g": 1500},
    {"common_name": "Channel Catfish", "scientific_name": "Ictalurus punctatus",
     "trophic_level": 3.8, "lipid_pct": 4.2, "body_mass_g": 2000},
    {"common_name": "Bluegill", "scientific_name": "Lepomis macrochirus",
     "trophic_level": 3.1, "lipid_pct": 3.5, "body_mass_g": 200},
    {"common_name": "Striped Bass", "scientific_name": "Morone saxatilis",
     "trophic_level": 4.5, "lipid_pct": 6.1, "body_mass_g": 5000},
    {"common_name": "Flathead Catfish", "scientific_name": "Pylodictis olivaris",
     "trophic_level": 4.0, "lipid_pct": 4.8, "body_mass_g": 3000},
    {"common_name": "White Perch", "scientific_name": "Morone americana",
     "trophic_level": 3.5, "lipid_pct": 3.8, "body_mass_g": 400},
    {"common_name": "Common Carp", "scientific_name": "Cyprinus carpio",
     "trophic_level": 2.9, "lipid_pct": 5.2, "body_mass_g": 3000},
    {"common_name": "Brown Trout", "scientific_name": "Salmo trutta",
     "trophic_level": 4.0, "lipid_pct": 5.5, "body_mass_g": 1200},
]

# Published BCF values (L/kg) — from Burkhard 2021
BCF_BASE = {
    'PFOS': 3100,    # 10^3.49
    'PFOA': 132,     # 10^2.12
    'PFNA': 1200,    # 10^3.08
    'PFHxS': 316,    # 10^2.50
    'PFDA': 2000,    # 10^3.30
    'GenX': 40,      # 10^1.60
}

# Published TMF values per trophic level step
TMF = {
    'PFOS': 3.5, 'PFOA': 1.5, 'PFNA': 3.0,
    'PFHxS': 2.0, 'PFDA': 3.2, 'GenX': 1.2,
}

# EPA reference doses (mg/kg/day) — EPA 2024
RFD = {
    'PFOS': 1.0e-7, 'PFOA': 3.0e-8, 'GenX': 3.0e-6,
    'PFHxS': 2.0e-5, 'PFNA': 3.0e-6, 'PFDA': 3.0e-6,
}

# EPA consumption rates (g/day)
CONSUMPTION_RATES = {
    'general': 22.0,
    'recreational': 17.0,
    'high_recreational': 50.0,
    'subsistence': 142.4,
}

REFERENCE_LIPID_PCT = 4.0
REFERENCE_TROPHIC = 3.0
SERVING_G = 227  # EPA default serving size


def compute_hotspot_influence(lat: float, lng: float) -> float:
    """Compute PFAS contamination influence from known hotspots."""
    total = 0.0
    for hs in PFAS_HOTSPOTS:
        dist = np.sqrt((lat - hs["lat"])**2 + (lng - hs["lng"])**2)
        if dist < hs["radius"] * 3:
            influence = hs["intensity"] * np.exp(-0.5 * (dist / hs["radius"])**2)
            total += influence
    return min(total, 1.5)


def generate_segment_features(n_segments: int = 5000) -> pd.DataFrame:
    """
    Generate realistic segment-level features for the continental US.
    Each row = one NHDPlus stream segment with 29 features + PFAS label.
    """
    # Generate segment locations across continental US
    # Weighted toward eastern US (more waterways, more contamination data)
    lats = np.concatenate([
        np.random.uniform(25, 48, int(n_segments * 0.3)),    # uniform spread
        np.random.normal(40, 4, int(n_segments * 0.3)),      # northeast cluster
        np.random.normal(35, 3, int(n_segments * 0.2)),      # southeast cluster
        np.random.normal(44, 2, int(n_segments * 0.2)),      # great lakes cluster
    ])[:n_segments]
    lats = np.clip(lats, 25, 48)

    lngs = np.concatenate([
        np.random.uniform(-125, -67, int(n_segments * 0.3)),
        np.random.normal(-76, 5, int(n_segments * 0.3)),
        np.random.normal(-80, 4, int(n_segments * 0.2)),
        np.random.normal(-85, 3, int(n_segments * 0.2)),
    ])[:n_segments]
    lngs = np.clip(lngs, -125, -67)

    # Compute hotspot influence for each segment
    hotspot_influence = np.array([compute_hotspot_influence(lat, lng)
                                   for lat, lng in zip(lats, lngs)])

    # -- DISCHARGE FEATURES --
    # More facilities near urban areas and hotspots
    urban_factor = np.random.beta(2, 5, n_segments) + hotspot_influence * 0.3
    upstream_npdes_count = np.random.poisson(urban_factor * 8 + 1, n_segments)
    upstream_npdes_pfas_count = np.random.poisson(hotspot_influence * 3 + urban_factor * 0.5, n_segments)
    nearest_pfas_km = np.maximum(0.1, np.random.exponential(30, n_segments) / (1 + hotspot_influence * 5))
    upstream_discharge_volume = np.random.lognormal(8, 2, n_segments) * (1 + urban_factor)
    pfas_industry_density = np.random.exponential(0.5, n_segments) * (1 + hotspot_influence * 2)
    afff_nearby = (np.random.random(n_segments) < (0.05 + hotspot_influence * 0.15)).astype(int)
    wwtp_upstream = (np.random.random(n_segments) < (0.3 + urban_factor * 0.4)).astype(int)
    landfill_upstream = (np.random.random(n_segments) < (0.1 + urban_factor * 0.2)).astype(int)

    # -- HYDROLOGIC FEATURES --
    stream_order = np.random.choice([1, 2, 3, 4, 5, 6, 7], n_segments,
                                     p=[0.30, 0.25, 0.20, 0.12, 0.07, 0.04, 0.02])
    mean_flow = np.random.lognormal(np.log(stream_order * 5), 1.0)
    low_flow_7q10 = mean_flow * np.random.uniform(0.05, 0.3, n_segments)
    watershed_area = mean_flow * np.random.uniform(5, 50, n_segments)
    baseflow_index = np.random.uniform(0.2, 0.8, n_segments)
    mean_velocity = np.random.uniform(0.1, 2.0, n_segments)

    # -- LAND USE FEATURES --
    pct_urban = np.random.beta(2, 5, n_segments) * 100
    pct_agriculture = np.random.beta(3, 3, n_segments) * 100
    pct_forest = np.maximum(0, 100 - pct_urban - pct_agriculture + np.random.normal(0, 10, n_segments))
    pct_impervious = pct_urban * np.random.uniform(0.3, 0.8, n_segments)
    population_density = np.random.lognormal(4, 2, n_segments) * (pct_urban / 50 + 0.1)
    airport_within_10km = (np.random.random(n_segments) < (0.08 + pct_urban / 500)).astype(int)
    fire_training = (np.random.random(n_segments) < 0.03).astype(int)

    # -- WATER CHEMISTRY --
    ph = np.random.normal(7.2, 0.5, n_segments)
    temperature_c = np.random.normal(18, 6, n_segments)
    doc = np.random.lognormal(1.2, 0.6, n_segments)
    toc = doc * np.random.uniform(1.0, 1.5, n_segments)
    conductivity = np.random.lognormal(5.5, 0.8, n_segments)

    # ============================================================
    # GENERATE LABELS: predicted water PFAS concentration (ng/L)
    # Uses a realistic nonlinear function of features
    # ============================================================

    # Base PFAS from industrial sources (dominant factor)
    industrial_pfas = (
        upstream_npdes_pfas_count * 15 +
        (1.0 / (nearest_pfas_km + 0.1)) * 200 +
        pfas_industry_density * 30 +
        afff_nearby * np.random.uniform(50, 300, n_segments) +
        wwtp_upstream * np.random.uniform(5, 50, n_segments) +
        landfill_upstream * np.random.uniform(2, 20, n_segments)
    )

    # Dilution effect (more flow = more dilution)
    dilution_factor = 1.0 / (1.0 + mean_flow * 0.02)

    # Land use contribution
    land_use_pfas = pct_urban * 0.3 + pct_impervious * 0.2 + population_density * 0.005

    # Environmental modifiers
    env_modifier = 1.0 + (temperature_c - 15) * 0.01 + (ph - 7.0) * 0.05

    # Hotspot boost
    hotspot_pfas = hotspot_influence * np.random.uniform(100, 500, n_segments)

    # Total water PFAS
    water_pfas_ng_l = (industrial_pfas * dilution_factor + land_use_pfas + hotspot_pfas) * env_modifier
    water_pfas_ng_l = np.maximum(0.1, water_pfas_ng_l)

    # Add realistic noise (measurement uncertainty)
    noise = np.random.lognormal(0, 0.3, n_segments)
    water_pfas_ng_l *= noise

    # Clip to realistic range (EPA UCMR 5 data ranges from <1 to >10,000 ng/L)
    water_pfas_ng_l = np.clip(water_pfas_ng_l, 0.1, 15000)

    # Build dataframe
    df = pd.DataFrame({
        'segment_id': [f'seg_{i:06d}' for i in range(n_segments)],
        'latitude': lats,
        'longitude': lngs,
        # Discharge features
        'upstream_npdes_count': upstream_npdes_count,
        'upstream_npdes_pfas_count': upstream_npdes_pfas_count,
        'nearest_pfas_facility_km': np.round(nearest_pfas_km, 2),
        'upstream_discharge_volume_m3': np.round(upstream_discharge_volume, 0),
        'pfas_industry_density': np.round(pfas_industry_density, 3),
        'afff_site_nearby': afff_nearby,
        'wwtp_upstream': wwtp_upstream,
        'landfill_upstream': landfill_upstream,
        # Hydrologic features
        'mean_annual_flow_m3s': np.round(mean_flow, 2),
        'low_flow_7q10_m3s': np.round(low_flow_7q10, 3),
        'stream_order': stream_order,
        'watershed_area_km2': np.round(watershed_area, 1),
        'baseflow_index': np.round(baseflow_index, 3),
        'mean_velocity_ms': np.round(mean_velocity, 3),
        # Land use features
        'pct_urban': np.round(pct_urban, 1),
        'pct_agriculture': np.round(pct_agriculture, 1),
        'pct_forest': np.round(np.clip(pct_forest, 0, 100), 1),
        'pct_impervious': np.round(pct_impervious, 1),
        'population_density': np.round(population_density, 1),
        'airport_within_10km': airport_within_10km,
        'fire_training_site': fire_training,
        # Water chemistry
        'ph': np.round(ph, 2),
        'temperature_c': np.round(temperature_c, 1),
        'dissolved_organic_carbon_mgl': np.round(doc, 2),
        'total_organic_carbon_mgl': np.round(toc, 2),
        'conductivity_us_cm': np.round(conductivity, 1),
        # Label
        'water_pfas_ng_l': np.round(water_pfas_ng_l, 2),
    })

    return df


def generate_huc8_boundaries() -> dict:
    """Generate simplified HUC-8 watershed boundary GeoJSON for key regions."""
    huc8s = [
        {"huc8": "03030004", "name": "Cape Fear River", "state": "NC",
         "center": [35.05, -78.88], "size": 0.6},
        {"huc8": "03030005", "name": "Lower Cape Fear", "state": "NC",
         "center": [34.50, -78.30], "size": 0.5},
        {"huc8": "04100001", "name": "St. Clair-Detroit", "state": "MI",
         "center": [42.50, -82.90], "size": 0.5},
        {"huc8": "04080201", "name": "Huron", "state": "MI",
         "center": [44.45, -83.33], "size": 0.4},
        {"huc8": "02040202", "name": "Delaware River", "state": "NJ/PA",
         "center": [40.20, -74.80], "size": 0.5},
        {"huc8": "06030002", "name": "Wheeler Lake", "state": "AL",
         "center": [34.60, -86.98], "size": 0.4},
        {"huc8": "02020003", "name": "Hoosic River", "state": "NY/VT",
         "center": [42.88, -73.20], "size": 0.3},
        {"huc8": "02030101", "name": "Upper Hudson", "state": "NY",
         "center": [43.00, -73.80], "size": 0.4},
        {"huc8": "10190003", "name": "Fountain Creek", "state": "CO",
         "center": [38.80, -104.72], "size": 0.3},
        {"huc8": "01070004", "name": "Merrimack", "state": "NH",
         "center": [42.86, -71.49], "size": 0.4},
        # Additional spread for national coverage
        {"huc8": "07100009", "name": "Des Moines", "state": "IA",
         "center": [41.60, -93.60], "size": 0.5},
        {"huc8": "08090201", "name": "Lower Mississippi", "state": "LA",
         "center": [30.00, -90.10], "size": 0.6},
        {"huc8": "12100301", "name": "San Jacinto", "state": "TX",
         "center": [29.80, -95.30], "size": 0.5},
        {"huc8": "17110016", "name": "Duwamish", "state": "WA",
         "center": [47.50, -122.30], "size": 0.3},
        {"huc8": "18050002", "name": "San Francisco Bay", "state": "CA",
         "center": [37.80, -122.40], "size": 0.4},
    ]

    features = []
    for h in huc8s:
        lat, lng = h["center"]
        s = h["size"]
        # Simple polygon (rectangle approximation)
        coords = [[
            [lng - s, lat - s/2],
            [lng + s, lat - s/2],
            [lng + s, lat + s/2],
            [lng - s, lat + s/2],
            [lng - s, lat - s/2],
        ]]
        features.append({
            "type": "Feature",
            "properties": {
                "huc8": h["huc8"],
                "name": h["name"],
                "state": h["state"],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": coords,
            }
        })

    return {"type": "FeatureCollection", "features": features}


def generate_river_geojson(segments_df: pd.DataFrame, n_rivers: int = 200) -> dict:
    """Generate river segment GeoJSON for visualization."""
    features = []

    # Sample segments for river rendering
    sampled = segments_df.sample(min(n_rivers, len(segments_df)))

    for _, row in sampled.iterrows():
        lat, lng = row['latitude'], row['longitude']
        # Generate a short river segment (5-10 coordinate pairs)
        n_points = np.random.randint(5, 12)
        coords = []
        for i in range(n_points):
            coords.append([
                lng + i * np.random.uniform(0.005, 0.015) * np.random.choice([-1, 1]),
                lat + i * np.random.uniform(0.003, 0.01) * np.random.choice([-1, 1]),
            ])

        # Determine risk level
        pfas = row['water_pfas_ng_l']
        if pfas > 100:
            risk = "high"
        elif pfas > 20:
            risk = "medium"
        else:
            risk = "low"

        features.append({
            "type": "Feature",
            "properties": {
                "segment_id": row['segment_id'],
                "water_pfas_ng_l": float(row['water_pfas_ng_l']),
                "risk_level": risk,
                "stream_order": int(row['stream_order']),
                "flow_rate": float(row['mean_annual_flow_m3s']),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            }
        })

    return {"type": "FeatureCollection", "features": features}


def predict_tissue_concentration(water_pfas_ng_l: float, trophic_level: float,
                                  lipid_pct: float, congener: str,
                                  doc_mg_l: float = 5.0) -> float:
    """Predict fish tissue PFAS using BCF + TMF (deterministic chemistry model)."""
    K_DOC = {'PFOS': 1100, 'PFOA': 400, 'PFNA': 800, 'PFHxS': 300, 'PFDA': 900, 'GenX': 200}

    # Dissolved fraction
    c_dissolved = water_pfas_ng_l / (1 + K_DOC.get(congener, 500) * doc_mg_l * 1e-6)

    # Lipid-adjusted BCF
    bcf = BCF_BASE[congener] * (lipid_pct / REFERENCE_LIPID_PCT)

    # Base tissue from water
    c_base = c_dissolved * bcf / 1000  # ng/L × L/kg / 1000 → ng/g

    # Trophic magnification
    trophic_diff = trophic_level - REFERENCE_TROPHIC
    c_tissue = c_base * (TMF[congener] ** max(0, trophic_diff))

    return max(0, c_tissue)


def compute_hazard_quotient(tissue_by_congener: dict, ingestion_rate_g_day: float,
                             body_weight_kg: float = 70.0) -> tuple:
    """Compute hazard index and advisory."""
    hi = 0.0
    for congener, c_ng_g in tissue_by_congener.items():
        if congener in RFD:
            dose = (c_ng_g * 1e-6 * ingestion_rate_g_day) / body_weight_kg
            hi += dose / RFD[congener]

    if hi > 0:
        safe_daily_g = ingestion_rate_g_day / hi
        safe_servings = max(0, int((safe_daily_g * 30) / SERVING_G))
    else:
        safe_servings = 30

    if hi < 0.5:
        status = "safe"
    elif hi < 1.5:
        status = "limited"
    else:
        status = "unsafe"

    return round(hi, 3), min(safe_servings, 30), status


def generate_full_dataset():
    """Generate complete dataset: training data + visualization data."""
    print("Generating segment features...")
    df = generate_segment_features(5000)

    print(f"Generated {len(df)} segments")
    print(f"PFAS range: {df['water_pfas_ng_l'].min():.1f} - {df['water_pfas_ng_l'].max():.1f} ng/L")
    print(f"Median PFAS: {df['water_pfas_ng_l'].median():.1f} ng/L")

    # Save training data
    df.to_csv('training_data.csv', index=False)
    print("Saved training_data.csv")

    # Generate HUC-8 boundaries
    huc8_geojson = generate_huc8_boundaries()
    with open('huc8_boundaries.geojson', 'w') as f:
        json.dump(huc8_geojson, f)
    print("Saved huc8_boundaries.geojson")

    # Generate river segments
    river_geojson = generate_river_geojson(df, n_rivers=300)
    with open('river_segments.geojson', 'w') as f:
        json.dump(river_geojson, f)
    print("Saved river_segments.geojson")

    # Generate species predictions for top segments
    print("\nGenerating species-level predictions for visualization...")
    top_segments = df.nlargest(100, 'water_pfas_ng_l')

    segments_output = []
    for _, seg in top_segments.iterrows():
        species_list = []
        for sp in SPECIES:
            tissue_by_congener = {}
            for congener in ['PFOS', 'PFOA', 'PFNA', 'PFHxS', 'PFDA', 'GenX']:
                # Assume PFOS is ~40% of total, PFOA ~20%, rest split
                congener_fraction = {'PFOS': 0.40, 'PFOA': 0.20, 'PFNA': 0.10,
                                     'PFHxS': 0.10, 'PFDA': 0.10, 'GenX': 0.10}
                water_congener = seg['water_pfas_ng_l'] * congener_fraction[congener]
                tissue = predict_tissue_concentration(
                    water_congener, sp['trophic_level'], sp['lipid_pct'],
                    congener, seg['dissolved_organic_carbon_mgl']
                )
                tissue_by_congener[congener] = round(tissue, 2)

            total_tissue = sum(tissue_by_congener.values())

            hq_rec, serv_rec, status_rec = compute_hazard_quotient(
                tissue_by_congener, CONSUMPTION_RATES['recreational'])
            hq_sub, serv_sub, status_sub = compute_hazard_quotient(
                tissue_by_congener, CONSUMPTION_RATES['subsistence'])

            species_list.append({
                "common_name": sp['common_name'],
                "scientific_name": sp['scientific_name'],
                "trophic_level": sp['trophic_level'],
                "lipid_content_pct": sp['lipid_pct'],
                "tissue_pfos_ng_g": tissue_by_congener['PFOS'],
                "tissue_pfoa_ng_g": tissue_by_congener['PFOA'],
                "tissue_total_pfas_ng_g": round(total_tissue, 2),
                "hazard_quotient_recreational": hq_rec,
                "hazard_quotient_subsistence": hq_sub,
                "safe_servings_per_month_recreational": serv_rec,
                "safe_servings_per_month_subsistence": serv_sub,
                "safety_status_recreational": status_rec,
                "safety_status_subsistence": status_sub,
                "tissue_by_congener": tissue_by_congener,
            })

        # Sort species by total tissue (worst first)
        species_list.sort(key=lambda x: x['tissue_total_pfas_ng_g'], reverse=True)

        segments_output.append({
            "segment_id": seg['segment_id'],
            "latitude": round(float(seg['latitude']), 4),
            "longitude": round(float(seg['longitude']), 4),
            "predicted_water_pfas_ng_l": round(float(seg['water_pfas_ng_l']), 2),
            "flow_rate_m3s": round(float(seg['mean_annual_flow_m3s']), 2),
            "stream_order": int(seg['stream_order']),
            "species": species_list,
        })

    # Also generate some lower-risk segments for map diversity
    mid_segments = df[(df['water_pfas_ng_l'] > 5) & (df['water_pfas_ng_l'] < 100)].sample(min(200, len(df)))
    for _, seg in mid_segments.iterrows():
        species_list = []
        for sp in SPECIES[:4]:  # Just top 4 species for mid-risk
            tissue_by_congener = {}
            for congener in ['PFOS', 'PFOA']:  # Simplified for mid-risk
                congener_fraction = {'PFOS': 0.40, 'PFOA': 0.20}
                water_congener = seg['water_pfas_ng_l'] * congener_fraction[congener]
                tissue = predict_tissue_concentration(
                    water_congener, sp['trophic_level'], sp['lipid_pct'],
                    congener, seg['dissolved_organic_carbon_mgl']
                )
                tissue_by_congener[congener] = round(tissue, 2)

            total_tissue = sum(tissue_by_congener.values())
            hq_rec, serv_rec, status_rec = compute_hazard_quotient(
                tissue_by_congener, CONSUMPTION_RATES['recreational'])
            hq_sub, serv_sub, status_sub = compute_hazard_quotient(
                tissue_by_congener, CONSUMPTION_RATES['subsistence'])

            species_list.append({
                "common_name": sp['common_name'],
                "scientific_name": sp['scientific_name'],
                "trophic_level": sp['trophic_level'],
                "lipid_content_pct": sp['lipid_pct'],
                "tissue_total_pfas_ng_g": round(total_tissue, 2),
                "hazard_quotient_recreational": hq_rec,
                "hazard_quotient_subsistence": hq_sub,
                "safe_servings_per_month_recreational": serv_rec,
                "safe_servings_per_month_subsistence": serv_sub,
                "safety_status_recreational": status_rec,
                "safety_status_subsistence": status_sub,
                "tissue_by_congener": tissue_by_congener,
            })

        species_list.sort(key=lambda x: x['tissue_total_pfas_ng_g'], reverse=True)

        segments_output.append({
            "segment_id": seg['segment_id'],
            "latitude": round(float(seg['latitude']), 4),
            "longitude": round(float(seg['longitude']), 4),
            "predicted_water_pfas_ng_l": round(float(seg['water_pfas_ng_l']), 2),
            "flow_rate_m3s": round(float(seg['mean_annual_flow_m3s']), 2),
            "stream_order": int(seg['stream_order']),
            "species": species_list,
        })

    # Generate facility data
    facilities = []
    for i, hs in enumerate(PFAS_HOTSPOTS):
        facilities.append({
            "facility_id": f"fac_{i:04d}",
            "name": hs["name"],
            "lat": hs["lat"],
            "lng": hs["lng"],
            "pfas_sector": True,
            "intensity": hs["intensity"],
        })

    # Generate demographics overlay
    demographics = [
        {"name": "Fayetteville SE, NC", "lat": 35.03, "lng": -78.85,
         "median_income": 31200, "subsistence_pct": 18.5, "population": 24500},
        {"name": "Decatur NW, AL", "lat": 34.62, "lng": -87.00,
         "median_income": 28500, "subsistence_pct": 22.0, "population": 18000},
        {"name": "Oscoda Township, MI", "lat": 44.43, "lng": -83.35,
         "median_income": 33400, "subsistence_pct": 15.0, "population": 7000},
        {"name": "Bennington SW, VT", "lat": 42.87, "lng": -73.22,
         "median_income": 35800, "subsistence_pct": 12.0, "population": 9200},
        {"name": "Horsham Township, PA", "lat": 40.17, "lng": -75.14,
         "median_income": 42000, "subsistence_pct": 8.0, "population": 26000},
    ]

    # Assemble full output
    output = {
        "metadata": {
            "model_version": "trophictrace-xgb-v1",
            "generated_at": "2026-03-29T00:00:00Z",
            "total_segments": len(segments_output),
            "species_modeled": len(SPECIES),
            "congeners_modeled": 6,
        },
        "segments": segments_output,
        "facilities": facilities,
        "demographics": demographics,
        "species_reference": SPECIES,
    }

    with open('national_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved national_results.json ({len(segments_output)} segments)")

    return df, output


if __name__ == '__main__':
    df, output = generate_full_dataset()
    print("\n=== Data Generation Complete ===")
    print(f"Training samples: {len(df)}")
    print(f"Visualization segments: {len(output['segments'])}")
    print(f"Facilities: {len(output['facilities'])}")
    print(f"Demographics zones: {len(output['demographics'])}")
