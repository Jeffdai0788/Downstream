"""
TrophicTrace — Backend → Frontend Data Sync
Transforms national_results.json from backend format into the two files
the React frontend expects:
  1. nationalResults.json  — segment data with segment_id, latitude, longitude keys
  2. riverGeometry.json    — GeoJSON FeatureCollection with river LineStrings

Run after inference.py to update the visualization.
Usage: python sync_to_frontend.py
"""

import json
import os
import numpy as np
import math

from water_polygons import generate_all_water_body_points

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DATA_DIR = os.path.join(BACKEND_DIR, '..', 'trophictrace-viz', 'src', 'data')


def generate_tiny_segment(lat, lng, rng, length=0.002):
    """
    Generate a very short LineString (3-4 points, ~200m) that stays
    extremely close to the source coordinate. This prevents land bleed.
    """
    n_pts = rng.randint(3, 5)
    angle = rng.uniform(0, 2 * np.pi)
    coords = []
    cx, cy = lng, lat
    step = length / n_pts

    for i in range(n_pts):
        # Tiny steps with slight curve
        wobble = rng.uniform(-0.0003, 0.0003)
        coords.append([
            round(cx + wobble, 6),
            round(cy + wobble * 0.7, 6),
        ])
        cx += np.cos(angle) * step + rng.uniform(-0.0002, 0.0002)
        cy += np.sin(angle) * step + rng.uniform(-0.0002, 0.0002)

    return coords


def make_feature(coords, hotspot_id, name, pfas_ng_l, risk_level, stream_order, comid):
    """Build a single GeoJSON Feature."""
    return {
        "type": "Feature",
        "properties": {
            "id": f"{hotspot_id}_{comid}",
            "hotspot_id": hotspot_id,
            "hotspot_name": name,
            "pfas_ng_l": round(float(pfas_ng_l), 1),
            "risk_level": risk_level,
            "stream_order": stream_order,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
    }


def slugify(name):
    """Create a hotspot_id slug from a name."""
    s = name.lower().replace(' ', '_').replace('—', '_').replace(',', '').replace("'", '')
    return ''.join(c for c in s if c.isalnum() or c == '_')[:40]


def risk_from_pfas(pfas):
    if pfas > 40:
        return 'critical'
    elif pfas > 8:
        return 'moderate'
    return 'low'


def map_risk_level(risk):
    """Map backend risk levels to frontend expected values."""
    return {'high': 'critical', 'medium': 'moderate', 'low': 'low'}.get(risk, risk)


def sync():
    # Load backend output
    backend_path = os.path.join(BACKEND_DIR, 'national_results.json')
    if not os.path.exists(backend_path):
        print("ERROR: national_results.json not found. Run inference.py first.")
        return

    with open(backend_path) as f:
        backend = json.load(f)

    print(f"Loaded {len(backend['segments'])} segments from backend")

    # === Transform nationalResults.json ===
    frontend_segments = []
    for i, seg in enumerate(backend['segments']):
        frontend_seg = {
            "segment_id": f"seg_{i:04d}",
            "name": seg.get('name', f"Monitoring Site #{i}"),
            "latitude": seg.get('lat', seg.get('latitude', 0)),
            "longitude": seg.get('lng', seg.get('longitude', 0)),
            "predicted_water_pfas_ng_l": seg['predicted_water_pfas_ng_l'],
            "prediction_confidence": seg.get('prediction_confidence', 0.7),
            "flow_rate_m3s": seg.get('flow_rate_m3s', 50.0),
            "stream_order": seg.get('stream_order', 4),
            "risk_level": map_risk_level(seg.get('risk_level', 'low')),
            "top_contributing_features": seg.get('top_contributing_features', []),
            "species": [],
        }

        for sp in seg.get('species', []):
            pathway = sp.get('pathway', {})
            species_entry = {
                "common_name": sp['common_name'],
                "scientific_name": sp.get('scientific_name', ''),
                "trophic_level": sp.get('trophic_level', 3.0),
                "lipid_content_pct": sp.get('lipid_content_pct', 4.0),
                "tissue_pfos_ng_g": sp.get('tissue_pfos_ng_g', 0),
                "tissue_pfoa_ng_g": sp.get('tissue_pfoa_ng_g', 0),
                "tissue_total_pfas_ng_g": sp.get('tissue_total_pfas_ng_g', 0),
                "hazard_quotient_recreational": sp.get('hazard_quotient_recreational', 0),
                "hazard_quotient_subsistence": sp.get('hazard_quotient_subsistence', 0),
                "safe_servings_per_month_recreational": sp.get('safe_servings_per_month_recreational', 30),
                "safe_servings_per_month_subsistence": sp.get('safe_servings_per_month_subsistence', 30),
                "safety_status_recreational": sp.get('safety_status_recreational', 'safe'),
                "safety_status_subsistence": sp.get('safety_status_subsistence', 'safe'),
                "tissue_by_congener": sp.get('tissue_by_congener', {}),
                "confidence_interval": sp.get('confidence_interval', [0, 0]),
                "pathway": {
                    "source_facility": pathway.get('source_facility', 'Unknown'),
                    "source_distance_km": pathway.get('source_distance_km', 0),
                    "dilution_factor": pathway.get('dilution_factor', 1.0),
                    "water_concentration_ng_l": pathway.get('water_concentration_ng_l', 0),
                    "baf_applied": pathway.get('baf_applied', 0),
                    "bcf_applied": pathway.get('baf_applied', 0),
                    "tmf_applied": 1.0,
                    "tissue_concentration_ng_g": pathway.get('tissue_concentration_ng_g', 0),
                },
            }
            frontend_seg['species'].append(species_entry)

        frontend_segments.append(frontend_seg)

    demographics = backend.get('demographics', [])

    frontend_output = {
        "metadata": backend.get('metadata', {}),
        "segments": frontend_segments,
        "facilities": backend.get('facilities', []),
        "demographics": demographics,
        "species_reference": backend.get('species_reference', []),
        "geojson_segments": backend.get('geojson_segments', {}),
    }

    # === Generate riverGeometry.json ===
    # IMPORTANT: Keep geometries TINY to prevent land bleed.
    # Each segment gets a 3-4 point LineString within ~200m of the source point.
    # The heatmap layer does the visual work; lines are just for hover/click detection.
    geo_features = []
    rng = np.random.RandomState(42)

    for i, seg in enumerate(frontend_segments):
        lat = seg['latitude']
        lng = seg['longitude']
        pfas = seg['predicted_water_pfas_ng_l']
        risk = seg['risk_level']
        stream_order = seg.get('stream_order', 4)
        name = backend['segments'][i].get('name', f"Segment {seg['segment_id']}")
        comid = backend['segments'][i].get('comid', 8893800 + i)
        hotspot_id = slugify(name)

        # Very short segment — stays near the water body coordinate
        coords = generate_tiny_segment(lat, lng, rng, length=0.002)
        feat = make_feature(coords, hotspot_id, name, pfas, risk, stream_order, comid)
        geo_features.append(feat)

    print(f"  Generated {len(geo_features)} segment features")

    # === Add dense water body surface points from polygon grids ===
    # Points generated inside verified water body polygons — guaranteed on water.
    poly_rng = np.random.RandomState(42)
    water_body_data = generate_all_water_body_points(poly_rng)
    lake_comid = 9900000
    lake_seg_start = len(frontend_segments)

    for wb_id, wb_data in water_body_data.items():
        nice_name = wb_data['name']
        points = wb_data['points']  # list of (lat, lng, pfas)

        for j, (lat, lng, pfas) in enumerate(points):
            risk = risk_from_pfas(pfas)
            hotspot_id = f"{wb_id}_{j:03d}"
            comid = lake_comid + j
            pt_name = f"{nice_name} — Station {j+1}"

            coords = generate_tiny_segment(lat, lng, rng, length=0.003)
            feat = make_feature(coords, hotspot_id, pt_name, pfas, risk, 6, comid)
            geo_features.append(feat)

            lake_seg = {
                "segment_id": f"seg_{lake_seg_start + len(geo_features):04d}",
                "name": pt_name,
                "latitude": lat,
                "longitude": lng,
                "predicted_water_pfas_ng_l": pfas,
                "prediction_confidence": round(rng.uniform(0.6, 0.85), 2),
                "flow_rate_m3s": round(rng.uniform(10, 500), 2),
                "stream_order": 6,
                "risk_level": risk,
                "top_contributing_features": frontend_segments[0].get('top_contributing_features', []) if frontend_segments else [],
                "species": _generate_lake_species(pfas, wb_id, rng),
            }
            frontend_segments.append(lake_seg)

        lake_comid += 1000
        print(f"  Added {len(points)} {nice_name} water surface points (polygon grid)")

    # Update the output with lake segments
    frontend_output['segments'] = frontend_segments

    river_geojson = {
        "type": "FeatureCollection",
        "features": geo_features,
    }

    # === Write output files ===
    os.makedirs(FRONTEND_DATA_DIR, exist_ok=True)

    nr_path = os.path.join(FRONTEND_DATA_DIR, 'nationalResults.json')
    with open(nr_path, 'w') as f:
        json.dump(frontend_output, f)
    nr_size = os.path.getsize(nr_path) / 1024 / 1024
    print(f"\nWrote {nr_path} ({nr_size:.1f} MB, {len(frontend_segments)} segments)")

    rg_path = os.path.join(FRONTEND_DATA_DIR, 'riverGeometry.json')
    with open(rg_path, 'w') as f:
        json.dump(river_geojson, f)
    rg_size = os.path.getsize(rg_path) / 1024 / 1024
    print(f"Wrote {rg_path} ({rg_size:.1f} MB, {len(geo_features)} features)")

    risk_counts = {}
    for seg in frontend_segments:
        r = seg['risk_level']
        risk_counts[r] = risk_counts.get(r, 0) + 1
    print(f"\nRisk distribution: {risk_counts}")
    print(f"Total map features: {len(geo_features)}")


def _generate_lake_species(water_pfas, lake_name, rng):
    """Generate species data for a lake point using analytic BAF formula."""
    from generate_data import SPECIES, BAF_TABLE, RFD, CONSUMPTION_RATES, SERVING_G, get_field_baf, compute_hazard_quotient

    CONGENERS = ['PFOS', 'PFOA', 'PFNA', 'PFHxS', 'PFDA', 'GenX']
    FRACTIONS = {'PFOS': 0.40, 'PFOA': 0.20, 'PFNA': 0.10, 'PFHxS': 0.10, 'PFDA': 0.10, 'GenX': 0.10}

    # Great Lakes species subset
    lake_species_names = ['Largemouth Bass', 'Yellow Perch', 'Channel Catfish', 'Smallmouth Bass', 'Common Carp', 'White Sucker']
    lake_species = [sp for sp in SPECIES if sp['common_name'] in lake_species_names]

    results = []
    for sp in lake_species:
        tissue_by_congener = {}
        total_tissue = 0
        for cong in CONGENERS:
            wc = water_pfas * FRACTIONS[cong]
            baf = get_field_baf(cong, sp['trophic_level'])
            tissue = wc * baf / 1000.0 * rng.uniform(0.85, 1.15)
            tissue = max(0, tissue)
            tissue_by_congener[cong] = round(tissue, 2)
            total_tissue += tissue

        hq_rec, serv_rec, status_rec = compute_hazard_quotient(tissue_by_congener, CONSUMPTION_RATES['recreational'])
        hq_sub, serv_sub, status_sub = compute_hazard_quotient(tissue_by_congener, CONSUMPTION_RATES['subsistence'])

        results.append({
            "common_name": sp['common_name'],
            "scientific_name": sp['scientific_name'],
            "trophic_level": sp['trophic_level'],
            "lipid_content_pct": sp['lipid_pct'],
            "tissue_pfos_ng_g": tissue_by_congener.get('PFOS', 0),
            "tissue_pfoa_ng_g": tissue_by_congener.get('PFOA', 0),
            "tissue_total_pfas_ng_g": round(total_tissue, 2),
            "hazard_quotient_recreational": hq_rec,
            "hazard_quotient_subsistence": hq_sub,
            "safe_servings_per_month_recreational": serv_rec,
            "safe_servings_per_month_subsistence": serv_sub,
            "safety_status_recreational": status_rec,
            "safety_status_subsistence": status_sub,
            "tissue_by_congener": tissue_by_congener,
            "confidence_interval": [round(total_tissue * 0.78, 2), round(total_tissue * 1.28, 2)],
            "pathway": {
                "source_facility": "WWTP effluent / atmospheric deposition",
                "source_distance_km": 0,
                "dilution_factor": 1.0,
                "water_concentration_ng_l": round(water_pfas, 2),
                "baf_applied": round(get_field_baf('PFOS', sp['trophic_level']), 0),
                "bcf_applied": round(get_field_baf('PFOS', sp['trophic_level']), 0),
                "tmf_applied": 1.0,
                "tissue_concentration_ng_g": round(total_tissue, 2),
            },
        })

    results.sort(key=lambda x: x['tissue_total_pfas_ng_g'], reverse=True)
    return results


if __name__ == '__main__':
    sync()
