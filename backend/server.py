"""
TrophicTrace — FastAPI Backend Server
Serves precomputed predictions + real-time PINN queries.
"""

import json
import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="TrophicTrace API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load precomputed results
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(DATA_DIR, 'national_results.json')) as f:
    NATIONAL_DATA = json.load(f)

with open(os.path.join(DATA_DIR, 'huc8_boundaries.geojson')) as f:
    HUC8_GEOJSON = json.load(f)

with open(os.path.join(DATA_DIR, 'training_metrics.json')) as f:
    XGB_METRICS = json.load(f)

with open(os.path.join(DATA_DIR, 'pinn_model_info.json')) as f:
    PINN_METRICS = json.load(f)


@app.get("/api/metadata")
def get_metadata():
    """Return model metadata and performance metrics."""
    return NATIONAL_DATA["metadata"]


@app.get("/api/segments")
def get_segments(
    min_lat: float = Query(None),
    max_lat: float = Query(None),
    min_lng: float = Query(None),
    max_lng: float = Query(None),
    risk_level: str = Query(None),
):
    """Return segments, optionally filtered by bounding box or risk level."""
    segments = NATIONAL_DATA["segments"]

    if min_lat is not None:
        segments = [s for s in segments if s["latitude"] >= min_lat]
    if max_lat is not None:
        segments = [s for s in segments if s["latitude"] <= max_lat]
    if min_lng is not None:
        segments = [s for s in segments if s["longitude"] >= min_lng]
    if max_lng is not None:
        segments = [s for s in segments if s["longitude"] <= max_lng]
    if risk_level:
        segments = [s for s in segments if s["risk_level"] == risk_level]

    return {"count": len(segments), "segments": segments}


@app.get("/api/segment/{segment_id}")
def get_segment(segment_id: str):
    """Return detailed data for a single segment."""
    for seg in NATIONAL_DATA["segments"]:
        if seg["segment_id"] == segment_id:
            return seg
    return JSONResponse(status_code=404, content={"error": "Segment not found"})


@app.get("/api/facilities")
def get_facilities():
    """Return all known PFAS facilities."""
    return {"facilities": NATIONAL_DATA["facilities"]}


@app.get("/api/demographics")
def get_demographics():
    """Return environmental justice demographic zones."""
    return {"demographics": NATIONAL_DATA["demographics"]}


@app.get("/api/geojson/segments")
def get_geojson_segments():
    """Return river segment GeoJSON for map rendering."""
    return NATIONAL_DATA.get("geojson_segments", {"type": "FeatureCollection", "features": []})


@app.get("/api/geojson/huc8")
def get_geojson_huc8():
    """Return HUC-8 watershed boundary GeoJSON."""
    return HUC8_GEOJSON


@app.get("/api/species")
def get_species():
    """Return reference species data."""
    return {"species": NATIONAL_DATA["species_reference"]}


@app.get("/api/metrics")
def get_metrics():
    """Return model performance metrics for display."""
    return {
        "xgboost": XGB_METRICS,
        "pinn": PINN_METRICS,
    }


@app.get("/api/summary")
def get_summary():
    """Return national summary statistics."""
    segments = NATIONAL_DATA["segments"]
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    total_unsafe_species = 0
    max_hq = 0

    for seg in segments:
        risk_counts[seg["risk_level"]] += 1
        for sp in seg["species"]:
            if sp["safety_status_subsistence"] == "unsafe":
                total_unsafe_species += 1
            max_hq = max(max_hq, sp.get("hazard_quotient_subsistence", 0))

    return {
        "total_segments": len(segments),
        "risk_distribution": risk_counts,
        "total_unsafe_species_instances": total_unsafe_species,
        "max_hazard_quotient_subsistence": round(max_hq, 2),
        "n_facilities": len(NATIONAL_DATA["facilities"]),
        "n_ej_zones": len(NATIONAL_DATA["demographics"]),
    }


# Serve static frontend files if they exist
FRONTEND_DIR = os.path.join(DATA_DIR, '..', 'trophictrace-viz', 'dist')
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
