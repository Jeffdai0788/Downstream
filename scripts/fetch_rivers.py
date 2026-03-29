"""
Fetch real river geometry from OpenStreetMap Overpass API for TrophicTrace hotspot regions.
- Only fetches NAMED rivers (not all waterways in bbox)
- Post-filters to keep geometry within proximity of data points
- Queries relations for major rivers (Ohio, Tennessee, etc.)
- Snaps data points to nearest geometry for accurate PFAS attribution
"""

import json
import math
import os
import ssl
import time
import urllib.request
import urllib.parse

ssl._create_default_https_context = ssl._create_unverified_context

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MAX_DISTANCE_KM = 5.0  # Only keep geometry within this distance of data points
SUBSEG_KM = 0.4

HOTSPOTS = [
    {
        "id": "cape_fear",
        "name": "Cape Fear River, NC",
        "rivers": ["Cape Fear River"],
        "bbox": [34.5, -79.2, 35.5, -78.5],
        "segments": [
            {"lat": 35.20, "lng": -78.98, "pfas": 180},
            {"lat": 35.15, "lng": -78.95, "pfas": 280},
            {"lat": 35.10, "lng": -78.92, "pfas": 420},
            {"lat": 35.05, "lng": -78.88, "pfas": 850},
            {"lat": 35.00, "lng": -78.86, "pfas": 920},
            {"lat": 34.95, "lng": -78.84, "pfas": 780},
            {"lat": 34.90, "lng": -78.80, "pfas": 600},
            {"lat": 34.85, "lng": -78.76, "pfas": 450},
            {"lat": 34.78, "lng": -78.72, "pfas": 320},
            {"lat": 34.70, "lng": -78.68, "pfas": 200},
        ],
    },
    {
        "id": "lake_michigan",
        "name": "Lake Michigan, Waukegan IL",
        "rivers": ["Waukegan River", "Dead River", "North Chicago Drainage Ditch"],
        "bbox": [42.30, -87.88, 42.40, -87.78],
        "segments": [
            {"lat": 42.38, "lng": -87.84, "pfas": 120},
            {"lat": 42.36, "lng": -87.82, "pfas": 280},
            {"lat": 42.34, "lng": -87.83, "pfas": 350},
            {"lat": 42.36, "lng": -87.86, "pfas": 200},
            {"lat": 42.33, "lng": -87.85, "pfas": 310},
            {"lat": 42.35, "lng": -87.80, "pfas": 180},
        ],
    },
    {
        "id": "ohio_river",
        "name": "Ohio River, Parkersburg WV",
        "rivers": ["Ohio River", "Little Kanawha River"],
        "use_relations": True,
        "bbox": [39.10, -81.70, 39.35, -81.35],
        "segments": [
            {"lat": 39.30, "lng": -81.60, "pfas": 400},
            {"lat": 39.28, "lng": -81.57, "pfas": 680},
            {"lat": 39.26, "lng": -81.55, "pfas": 1200},
            {"lat": 39.24, "lng": -81.53, "pfas": 1050},
            {"lat": 39.22, "lng": -81.50, "pfas": 800},
            {"lat": 39.20, "lng": -81.48, "pfas": 550},
            {"lat": 39.18, "lng": -81.45, "pfas": 380},
            {"lat": 39.16, "lng": -81.42, "pfas": 250},
        ],
    },
    {
        "id": "delaware_river",
        "name": "Delaware River, Bucks County PA",
        "rivers": ["Delaware River", "Neshaminy Creek"],
        "use_relations": True,
        "bbox": [39.98, -75.05, 40.25, -74.72],
        "segments": [
            {"lat": 40.22, "lng": -74.88, "pfas": 150},
            {"lat": 40.18, "lng": -74.85, "pfas": 320},
            {"lat": 40.15, "lng": -74.82, "pfas": 480},
            {"lat": 40.12, "lng": -74.80, "pfas": 520},
            {"lat": 40.08, "lng": -74.78, "pfas": 400},
            {"lat": 40.05, "lng": -74.76, "pfas": 280},
        ],
    },
    {
        "id": "huron_river",
        "name": "Huron River, Ann Arbor MI",
        "rivers": ["Huron River"],
        "bbox": [42.22, -83.85, 42.35, -83.65],
        "segments": [
            {"lat": 42.32, "lng": -83.80, "pfas": 100},
            {"lat": 42.30, "lng": -83.76, "pfas": 220},
            {"lat": 42.28, "lng": -83.74, "pfas": 380},
            {"lat": 42.27, "lng": -83.72, "pfas": 300},
            {"lat": 42.26, "lng": -83.70, "pfas": 180},
        ],
    },
    {
        "id": "merrimack_river",
        "name": "Merrimack River, NH",
        "rivers": ["Merrimack River"],
        "use_relations": True,
        "bbox": [42.75, -71.50, 42.92, -71.22],
        "segments": [
            {"lat": 42.88, "lng": -71.34, "pfas": 180},
            {"lat": 42.86, "lng": -71.32, "pfas": 350},
            {"lat": 42.84, "lng": -71.30, "pfas": 420},
            {"lat": 42.82, "lng": -71.28, "pfas": 360},
            {"lat": 42.80, "lng": -71.26, "pfas": 220},
        ],
    },
    {
        "id": "tennessee_river",
        "name": "Tennessee River, Decatur AL",
        "rivers": ["Tennessee River"],
        "use_relations": True,
        "bbox": [34.45, -87.10, 34.65, -86.85],
        "segments": [
            {"lat": 34.62, "lng": -87.02, "pfas": 300},
            {"lat": 34.60, "lng": -86.98, "pfas": 580},
            {"lat": 34.58, "lng": -86.96, "pfas": 900},
            {"lat": 34.56, "lng": -86.94, "pfas": 750},
            {"lat": 34.54, "lng": -86.92, "pfas": 500},
            {"lat": 34.52, "lng": -86.88, "pfas": 320},
        ],
    },
]


def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def min_distance_to_data(coord, segments):
    """Distance in km from a [lng, lat] coordinate to the nearest data point."""
    return min(haversine_km(coord[1], coord[0], s["lat"], s["lng"]) for s in segments)


def interpolate_pfas(lat, lng, segments):
    weights, values = [], []
    for seg in segments:
        d = max(haversine_km(lat, lng, seg["lat"], seg["lng"]), 0.01)
        w = 1.0 / (d ** 2)
        weights.append(w)
        values.append(seg["pfas"])
    return sum(w * v for w, v in zip(weights, values)) / sum(weights)


def query_overpass(query):
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(OVERPASS_URL, data=data, method="POST")
    req.add_header("User-Agent", "TrophicTrace/1.0")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"    Overpass error: {e}")
        return None


def overpass_to_linestrings(result):
    if not result or "elements" not in result:
        return []
    nodes = {}
    ways = []
    for el in result["elements"]:
        if el["type"] == "node":
            nodes[el["id"]] = [el["lon"], el["lat"]]
        elif el["type"] == "way":
            ways.append(el)
    lines = []
    for way in ways:
        coords = [nodes[nid] for nid in way.get("nodes", []) if nid in nodes]
        if len(coords) >= 2:
            lines.append(coords)
    return lines


def filter_lines_near_data(lines, segments, max_km):
    """Keep only lines where at least one coordinate is within max_km of a data point."""
    kept = []
    for line in lines:
        # Check if any point in the line is near a data point
        near = False
        for coord in line:
            if min_distance_to_data(coord, segments) <= max_km:
                near = True
                break
        if near:
            # Trim: only keep the portion of the line near data points
            trimmed = trim_line_near_data(line, segments, max_km)
            if len(trimmed) >= 2:
                kept.append(trimmed)
    return kept


def trim_line_near_data(line, segments, max_km):
    """Keep only the contiguous portion of a line that's within max_km of data."""
    # Find the longest contiguous run of coordinates near data
    runs = []
    current_run = []
    for coord in line:
        if min_distance_to_data(coord, segments) <= max_km * 1.5:
            current_run.append(coord)
        else:
            if len(current_run) >= 2:
                runs.append(current_run)
            current_run = []
    if len(current_run) >= 2:
        runs.append(current_run)

    if not runs:
        return line  # Keep whole line if nothing is near (shouldn't happen after filter)
    return max(runs, key=len)


def subsegment_line(coords):
    if len(coords) < 2:
        return [coords]
    subsegments = []
    current = [coords[0]]
    for i in range(1, len(coords)):
        current.append(coords[i])
        seg_len = sum(
            haversine_km(current[j][1], current[j][0], current[j + 1][1], current[j + 1][0])
            for j in range(len(current) - 1)
        )
        if seg_len >= SUBSEG_KM * 0.8:
            subsegments.append(current)
            current = [current[-1]]
    if len(current) >= 2:
        subsegments.append(current)
    return subsegments


def build_query(hotspot):
    """Build Overpass query — uses relations for major rivers, ways for smaller ones."""
    bbox = hotspot["bbox"]
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    use_rel = hotspot.get("use_relations", False)

    parts = []
    for river_name in hotspot["rivers"]:
        if use_rel:
            # Query both relations and ways for named rivers
            parts.append(f'relation["waterway"="river"]["name"="{river_name}"]({bbox_str});')
            parts.append(f'way["waterway"="river"]["name"="{river_name}"]({bbox_str});')
        else:
            parts.append(f'way["waterway"="river"]["name"="{river_name}"]({bbox_str});')

    # Join parts
    query_body = "\n  ".join(parts)
    return f"""
[out:json][timeout:45];
(
  {query_body}
);
out body;
>;
out skel qt;
"""


def generate_synthetic_river(hotspot):
    """Fallback: curved path through data point coordinates."""
    segs = sorted(hotspot["segments"], key=lambda s: -s["lat"])
    coords = [[s["lng"], s["lat"]] for s in segs]
    # Add curvature via midpoints
    interpolated = []
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        interpolated.append(p1)
        offset = ((hash(str(p1)) % 100) - 50) * 0.0002
        mid = [(p1[0] + p2[0]) / 2 + offset, (p1[1] + p2[1]) / 2 + offset * 0.5]
        interpolated.append(mid)
    interpolated.append(coords[-1])
    return [interpolated]


def process_hotspot(hotspot):
    """Fetch, filter, sub-segment, and attribute PFAS for one hotspot."""
    print(f"\n  Fetching {hotspot['name']}...")
    query = build_query(hotspot)
    result = query_overpass(query)
    lines = overpass_to_linestrings(result)
    print(f"    Raw: {len(lines)} way segments")

    if not lines:
        print(f"    No data — using synthetic fallback")
        lines = generate_synthetic_river(hotspot)
    else:
        # Filter to only keep geometry near data points
        before = len(lines)
        lines = filter_lines_near_data(lines, hotspot["segments"], MAX_DISTANCE_KM)
        print(f"    After proximity filter ({MAX_DISTANCE_KM}km): {len(lines)} lines (removed {before - len(lines)})")

    # Sub-segment and attribute PFAS
    features = []
    fid = 0
    for line in lines:
        for subseg in subsegment_line(line):
            if len(subseg) < 2:
                continue
            mid = subseg[len(subseg) // 2]
            pfas = interpolate_pfas(mid[1], mid[0], hotspot["segments"])

            if pfas < 200:
                risk = "low"
            elif pfas < 500:
                risk = "moderate"
            elif pfas < 900:
                risk = "high"
            else:
                risk = "critical"

            features.append({
                "type": "Feature",
                "properties": {
                    "id": f"{hotspot['id']}_{fid:04d}",
                    "hotspot_id": hotspot["id"],
                    "hotspot_name": hotspot["name"],
                    "pfas_ng_l": round(pfas, 1),
                    "risk_level": risk,
                    "stream_order": 5 if any(r in hotspot["rivers"][:1] for r in hotspot["rivers"]) else 3,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[round(c[0], 6), round(c[1], 6)] for c in subseg],
                },
            })
            fid += 1

    print(f"    Output: {len(features)} sub-segments")
    return features


def main():
    print("Fetching real river geometry from OpenStreetMap...\n")
    all_features = []

    for i, hotspot in enumerate(HOTSPOTS):
        features = process_hotspot(hotspot)
        all_features.extend(features)

        # Rate limit between requests
        if i < len(HOTSPOTS) - 1:
            print("    Waiting 15s for rate limit...")
            time.sleep(15)

    geojson = {"type": "FeatureCollection", "features": all_features}

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "trophictrace-viz", "src", "data", "riverGeometry.json"
    )
    out_path = os.path.abspath(out_path)

    with open(out_path, "w") as f:
        json.dump(geojson, f)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote {out_path}")
    print(f"  {len(all_features)} features, {size_kb:.0f} KB")

    # Summary
    from collections import Counter
    for hid, count in sorted(Counter(f["properties"]["hotspot_id"] for f in all_features).items()):
        feats = [f for f in all_features if f["properties"]["hotspot_id"] == hid]
        coords = sum(len(f["geometry"]["coordinates"]) for f in feats)
        pfas = [f["properties"]["pfas_ng_l"] for f in feats]
        print(f"  {hid}: {count} features, {coords} coords, PFAS {min(pfas):.0f}-{max(pfas):.0f}")


if __name__ == "__main__":
    main()
