#!/usr/bin/env python3
"""
TrophicTrace Real Environmental Data Fetcher
=============================================
Fetches items 3, 4, 5, 7, 8 from the data pipeline requirements:
  3. Upstream PFAS Facility Count + Distance (EPA TRI / ECHO)
  4. Stream Order + Watershed Area (NHDPlus via NLDI + NHDPLUSV2 WFS)
  5. Urban Land Cover % (EPA StreamCat)
  7. Seasonal Flow Rates (USGS NWIS + NHDPlus EROM)
  8. Fish Species Presence by Watershed (GBIF)

All results are saved to backend/real_data/ as individual JSON files,
then merged into a single enriched_environmental_data.json.
"""

import json, os, sys, time, math, traceback
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.parse
import urllib.error

BASE_DIR = Path(__file__).parent
REAL_DATA_DIR = BASE_DIR / "real_data"
REAL_DATA_DIR.mkdir(exist_ok=True)

TIMEOUT = 45  # seconds per API call

# ── Known PFAS hotspot coordinates (our monitoring stations) ──────────────────
# These are the 14 hotspots used in inference.py / generate_realistic_rivers.py
HOTSPOTS = [
    {"id": "cape_fear_nc",     "name": "Cape Fear River, NC (Chemours)",       "lat": 34.27, "lng": -78.95, "state": "NC"},
    {"id": "decatur_al",       "name": "Tennessee River, Decatur AL (3M)",     "lat": 34.60, "lng": -86.98, "state": "AL"},
    {"id": "oscoda_mi",        "name": "Au Sable River, MI (Wurtsmith AFB)",   "lat": 44.45, "lng": -83.33, "state": "MI"},
    {"id": "fayetteville_nc",  "name": "Cape Fear Basin, Fayetteville NC",     "lat": 35.05, "lng": -78.88, "state": "NC"},
    {"id": "parkersburg_wv",   "name": "Ohio River, Parkersburg WV (DuPont)",  "lat": 39.27, "lng": -81.54, "state": "WV"},
    {"id": "bennington_vt",    "name": "Walloomsac River, Bennington VT",      "lat": 42.88, "lng": -73.20, "state": "VT"},
    {"id": "rockford_mi",      "name": "Rogue River, Rockford MI (Wolverine)", "lat": 43.12, "lng": -85.56, "state": "MI"},
    {"id": "newburgh_ny",      "name": "Hudson River, Newburgh NY",            "lat": 41.50, "lng": -74.01, "state": "NY"},
    {"id": "warminster_pa",    "name": "Little Neshaminy Creek, PA (NASJRB)",  "lat": 40.20, "lng": -75.09, "state": "PA"},
    {"id": "colorado_springs",  "name": "Fountain Creek, CO (Peterson AFB)",   "lat": 38.75, "lng": -104.79, "state": "CO"},
    {"id": "great_lakes_mi",   "name": "Lake Huron Shoreline, MI",             "lat": 43.80, "lng": -83.00, "state": "MI"},
    {"id": "wilmington_nc",    "name": "Cape Fear Estuary, Wilmington NC",     "lat": 34.05, "lng": -77.95, "state": "NC"},
    {"id": "pease_nh",         "name": "Pease AFB, Portsmouth NH",             "lat": 43.08, "lng": -70.82, "state": "NH"},
    {"id": "paulsboro_nj",     "name": "Delaware River, Paulsboro NJ",        "lat": 39.83, "lng": -75.24, "state": "NJ"},
]

# ── 8 target fish species ─────────────────────────────────────────────────────
FISH_SPECIES = [
    {"common": "Largemouth Bass",   "scientific": "Micropterus salmoides",   "gbif_key": 2382397},
    {"common": "Channel Catfish",    "scientific": "Ictalurus punctatus",     "gbif_key": 2341144},
    {"common": "Striped Bass",       "scientific": "Morone saxatilis",        "gbif_key": 2385645},
    {"common": "Brook Trout",        "scientific": "Salvelinus fontinalis",   "gbif_key": 5202559},
    {"common": "Yellow Perch",       "scientific": "Perca flavescens",        "gbif_key": 2382763},
    {"common": "Common Carp",        "scientific": "Cyprinus carpio",         "gbif_key": 4286952},
    {"common": "Smallmouth Bass",    "scientific": "Micropterus dolomieu",    "gbif_key": 2382407},
    {"common": "White Sucker",       "scientific": "Catostomus commersonii",  "gbif_key": 2365034},
]


def fetch_json(url, label=""):
    """Fetch JSON from URL with timeout and error handling."""
    print(f"  [GET] {label or url[:80]}...")
    req = urllib.request.Request(url, headers={"User-Agent": "TrophicTrace/1.0 (YHack 2026 Research)"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"    ✓ Got response")
            return data
    except urllib.error.HTTPError as e:
        print(f"    ✗ HTTP {e.code}: {e.reason}")
        return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM 3: Upstream PFAS Facility Count + Distance
# Source: EPA ECHO (Enforcement & Compliance History Online)
#         EPA TRI (Toxics Release Inventory) via Envirofacts
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_pfas_facilities():
    """
    Fetch real PFAS-related facilities from EPA ECHO and TRI.
    For each hotspot, compute:
      - Number of PFAS facilities within 50km
      - Distance to nearest PFAS facility
    """
    print("\n" + "="*70)
    print("ITEM 3: Upstream PFAS Facility Count + Distance")
    print("="*70)

    all_facilities = []

    # ── EPA ECHO: PFAS-relevant SIC codes ──
    # 2821=Plastics, 2869=Industrial Organics, 4952=Sewerage, 9711=Military
    states = list(set(h["state"] for h in HOTSPOTS))
    sic_codes = ["2821", "2869", "4952", "9711"]

    for state in states:
        for sic in sic_codes:
            url = (
                f"https://echo.epa.gov/api/facilities?"
                f"output=JSON&p_st={state}&p_sic={sic}&p_act=Y"
                f"&responseset=500"
            )
            data = fetch_json(url, f"ECHO facilities SIC={sic} in {state}")
            if data and "Results" in data and "Facilities" in data["Results"]:
                for fac in data["Results"]["Facilities"]:
                    try:
                        lat = float(fac.get("Lat83", 0))
                        lng = float(fac.get("Lon83", 0))
                        if lat != 0 and lng != 0:
                            all_facilities.append({
                                "name": fac.get("FacName", "Unknown"),
                                "lat": lat,
                                "lng": lng,
                                "state": state,
                                "sic": sic,
                                "registry_id": fac.get("RegistryID", ""),
                                "npdes_id": fac.get("CWPName", ""),
                                "source": "EPA_ECHO",
                                "pfas_relevant_sic": True,
                            })
                    except (ValueError, TypeError):
                        continue
            time.sleep(0.3)  # rate limiting

    # ── EPA TRI: PFAS chemical releases ──
    # TRI tracks specific PFAS chemicals since 2020 reporting year
    pfas_chemicals = [
        "Perfluorooctanoic acid",
        "Perfluorooctane sulfonic acid",
        "GenX",
    ]
    for state in states:
        url = (
            f"https://data.epa.gov/efservice/TRI_FACILITY/"
            f"STATE_ABBR/{state}/rows/0:100/JSON"
        )
        data = fetch_json(url, f"TRI facilities in {state}")
        if data and isinstance(data, list):
            for fac in data:
                try:
                    lat = float(fac.get("LATITUDE", 0))
                    lng = float(fac.get("LONGITUDE", 0))
                    if lat != 0 and lng != 0:
                        all_facilities.append({
                            "name": fac.get("FACILITY_NAME", "Unknown"),
                            "lat": lat,
                            "lng": lng,
                            "state": state,
                            "sic": fac.get("PRIMARY_SIC", ""),
                            "registry_id": fac.get("TRI_FACILITY_ID", ""),
                            "source": "EPA_TRI",
                            "pfas_relevant_sic": str(fac.get("PRIMARY_SIC", "")) in sic_codes,
                        })
                except (ValueError, TypeError):
                    continue
        time.sleep(0.3)

    # Deduplicate by proximity (within 0.5km = same facility)
    unique = []
    for fac in all_facilities:
        is_dup = False
        for u in unique:
            if haversine_km(fac["lat"], fac["lng"], u["lat"], u["lng"]) < 0.5:
                is_dup = True
                break
        if not is_dup:
            unique.append(fac)

    print(f"\n  Total facilities fetched: {len(all_facilities)}")
    print(f"  After dedup: {len(unique)}")

    # ── Compute per-hotspot metrics ──
    hotspot_facility_metrics = []
    for hs in HOTSPOTS:
        nearby = []
        min_dist = float("inf")
        for fac in unique:
            d = haversine_km(hs["lat"], hs["lng"], fac["lat"], fac["lng"])
            if d < 50:  # within 50km radius
                nearby.append({**fac, "distance_km": round(d, 2)})
            if d < min_dist:
                min_dist = d

        hotspot_facility_metrics.append({
            "hotspot_id": hs["id"],
            "hotspot_name": hs["name"],
            "lat": hs["lat"],
            "lng": hs["lng"],
            "pfas_facility_count_50km": len(nearby),
            "nearest_pfas_facility_km": round(min_dist, 2) if min_dist < float("inf") else None,
            "nearby_facilities": sorted(nearby, key=lambda x: x["distance_km"])[:10],
        })

    result = {
        "source": "EPA ECHO + TRI",
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "total_unique_facilities": len(unique),
        "all_facilities": unique,
        "hotspot_metrics": hotspot_facility_metrics,
    }

    out = REAL_DATA_DIR / "pfas_facilities_real.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  ✓ Saved to {out} ({len(unique)} facilities)")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM 4: Stream Order + Watershed Area
# Source: USGS NLDI (Network Linked Data Index) → snaps lat/lon to COMID
#         NHDPlus attributes via NLDI feature lookup
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_stream_attributes():
    """
    For each hotspot, snap to the nearest NHDPlus COMID using NLDI,
    then retrieve stream order and watershed area from the NHDPlus VAA.
    """
    print("\n" + "="*70)
    print("ITEM 4: Stream Order + Watershed Area (NHDPlus via NLDI)")
    print("="*70)

    results = []

    for hs in HOTSPOTS:
        # Step 1: Use NLDI to find the nearest NHDPlus comid
        nldi_url = (
            f"https://labs.waterdata.usgs.gov/api/nldi/linked-data/comid/position?"
            f"coords=POINT({hs['lng']} {hs['lat']})"
        )
        comid_data = fetch_json(nldi_url, f"NLDI snap {hs['id']}")

        comid = None
        if comid_data and "features" in comid_data and len(comid_data["features"]) > 0:
            props = comid_data["features"][0].get("properties", {})
            comid = props.get("comid") or props.get("identifier")
            if comid:
                comid = str(comid)

        if not comid:
            print(f"    ⚠ No COMID found for {hs['id']}, trying alternate endpoint")
            # Alternate: use the hydro-network-linked-data endpoint
            nldi_url2 = (
                f"https://labs.waterdata.usgs.gov/api/nldi/linked-data/comid/position?"
                f"coords=POINT({hs['lng']}%20{hs['lat']})"
            )
            comid_data = fetch_json(nldi_url2, f"NLDI snap alt {hs['id']}")
            if comid_data and "features" in comid_data and len(comid_data["features"]) > 0:
                props = comid_data["features"][0].get("properties", {})
                comid = str(props.get("comid") or props.get("identifier") or "")

        # Step 2: Get NHDPlus flowline attributes via NLDI
        stream_order = None
        watershed_area_km2 = None
        reach_length_km = None

        if comid:
            # Try to get flowline attributes from NLDI
            attr_url = f"https://labs.waterdata.usgs.gov/api/nldi/linked-data/comid/{comid}"
            attr_data = fetch_json(attr_url, f"NLDI attributes COMID={comid}")
            if attr_data and "features" in attr_data and len(attr_data["features"]) > 0:
                props = attr_data["features"][0].get("properties", {})
                # NHDPlus VAA attributes that may be present
                stream_order = props.get("streamorde") or props.get("StreamOrde") or props.get("stream_order")
                watershed_area_km2 = props.get("totdasqkm") or props.get("TotDASqKM") or props.get("totda")
                reach_length_km = props.get("lengthkm") or props.get("LengthKM")

            # Step 3: Get basin (watershed) geometry for area calculation
            basin_url = f"https://labs.waterdata.usgs.gov/api/nldi/linked-data/comid/{comid}/basin"
            basin_data = fetch_json(basin_url, f"NLDI basin COMID={comid}")
            if basin_data and "features" in basin_data and len(basin_data["features"]) > 0:
                basin_props = basin_data["features"][0].get("properties", {})
                # Basin area might be in properties
                if not watershed_area_km2:
                    watershed_area_km2 = basin_props.get("TOT_BASIN_AREA") or basin_props.get("basin_area")

        # Step 4: Try EPA WATERS GeoServices for additional NHDPlus attributes
        if comid and (stream_order is None or watershed_area_km2 is None):
            waters_url = (
                f"https://watersgeo.epa.gov/arcgis/rest/services/NHDPlus_NP21/NHDSnapshot_NP21/"
                f"MapServer/0/query?where=COMID={comid}&outFields=StreamOrde,TotDASqKM,LengthKM,Fcode"
                f"&f=json"
            )
            waters_data = fetch_json(waters_url, f"EPA WATERS COMID={comid}")
            if waters_data and "features" in waters_data and len(waters_data["features"]) > 0:
                attrs = waters_data["features"][0].get("attributes", {})
                if stream_order is None:
                    stream_order = attrs.get("StreamOrde")
                if watershed_area_km2 is None:
                    watershed_area_km2 = attrs.get("TotDASqKM")
                if reach_length_km is None:
                    reach_length_km = attrs.get("LengthKM")

        results.append({
            "hotspot_id": hs["id"],
            "hotspot_name": hs["name"],
            "lat": hs["lat"],
            "lng": hs["lng"],
            "comid": comid,
            "stream_order": int(stream_order) if stream_order else None,
            "watershed_area_km2": float(watershed_area_km2) if watershed_area_km2 else None,
            "reach_length_km": float(reach_length_km) if reach_length_km else None,
        })
        time.sleep(0.4)  # rate limit

    filled = sum(1 for r in results if r["stream_order"] is not None)
    print(f"\n  Stream order filled: {filled}/{len(results)}")
    filled_ws = sum(1 for r in results if r["watershed_area_km2"] is not None)
    print(f"  Watershed area filled: {filled_ws}/{len(results)}")

    output = {
        "source": "USGS NLDI + NHDPlus + EPA WATERS",
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "hotspot_stream_attributes": results,
    }
    out = REAL_DATA_DIR / "stream_attributes_real.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ Saved to {out}")
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM 5: Urban Land Cover %
# Source: EPA StreamCat dataset (pre-computed NLCD metrics per COMID)
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_urban_landcover(stream_data):
    """
    Fetch urban land cover % from EPA StreamCat for each COMID.
    StreamCat pre-calculates NLCD metrics for every NHDPlus catchment.
    """
    print("\n" + "="*70)
    print("ITEM 5: Urban Land Cover % (EPA StreamCat)")
    print("="*70)

    results = []
    comids = []

    for attr in stream_data.get("hotspot_stream_attributes", []):
        if attr.get("comid"):
            comids.append(attr["comid"])

    if not comids:
        print("  ⚠ No COMIDs available, will use NLCD web service fallback")

    # StreamCat API endpoint
    # The API accepts comma-separated COMIDs
    comid_str = ",".join(comids[:20])  # batch up to 20

    if comid_str:
        # StreamCat REST API for NLCD land cover
        sc_url = (
            f"https://java.epa.gov/StreamCAT/metrics?"
            f"name=NLCD2019&comids={comid_str}&areaOfInterest=catchment,watershed"
        )
        sc_data = fetch_json(sc_url, f"StreamCat NLCD2019 for {len(comids)} COMIDs")

        if sc_data and isinstance(sc_data, list):
            for record in sc_data:
                comid = str(record.get("COMID", ""))
                results.append({
                    "comid": comid,
                    "pct_urban_catchment": record.get("PctUrbMd2019Cat", record.get("PCTURB2019CAT")),
                    "pct_urban_watershed": record.get("PctUrbMd2019Ws", record.get("PCTURB2019WS")),
                    "pct_impervious_catchment": record.get("PctImp2019Cat", record.get("PCTIMP2019CAT")),
                    "pct_impervious_watershed": record.get("PctImp2019Ws", record.get("PCTIMP2019WS")),
                    "pct_forest_catchment": record.get("PctDecid2019Cat"),
                    "pct_agriculture_catchment": record.get("PctCrop2019Cat"),
                    "source": "EPA_StreamCat",
                })
        else:
            print("  ⚠ StreamCat API did not return expected format, trying alternate...")
            # Try individual COMID queries
            for comid in comids[:14]:
                sc_url2 = (
                    f"https://java.epa.gov/StreamCAT/metrics?"
                    f"name=NLCD2019&comids={comid}&areaOfInterest=watershed"
                )
                sc_data2 = fetch_json(sc_url2, f"StreamCat COMID={comid}")
                if sc_data2 and isinstance(sc_data2, list) and len(sc_data2) > 0:
                    rec = sc_data2[0]
                    results.append({
                        "comid": comid,
                        "pct_urban_catchment": rec.get("PctUrbMd2019Cat"),
                        "pct_urban_watershed": rec.get("PctUrbMd2019Ws"),
                        "pct_impervious_catchment": rec.get("PctImp2019Cat"),
                        "pct_impervious_watershed": rec.get("PctImp2019Ws"),
                        "source": "EPA_StreamCat",
                    })
                time.sleep(0.5)

    # Fallback: use NLCD raster web service if StreamCat doesn't have it
    if len(results) < len(HOTSPOTS):
        print("  Supplementing with MRLC NLCD point query...")
        for hs in HOTSPOTS:
            already = any(r.get("comid") and r["comid"] in [a.get("comid") for a in stream_data.get("hotspot_stream_attributes", []) if a["hotspot_id"] == hs["id"]] for r in results)
            if not already:
                # MRLC (Multi-Resolution Land Characteristics) point query
                nlcd_url = (
                    f"https://www.mrlc.gov/api/v1/point?"
                    f"lat={hs['lat']}&lon={hs['lng']}&year=2019"
                )
                nlcd_data = fetch_json(nlcd_url, f"MRLC NLCD point {hs['id']}")
                if nlcd_data:
                    lc_class = nlcd_data.get("land_cover_class", "")
                    is_urban = "Developed" in str(lc_class)
                    results.append({
                        "hotspot_id": hs["id"],
                        "land_cover_class": lc_class,
                        "is_urban": is_urban,
                        "source": "MRLC_NLCD_point",
                    })
                time.sleep(0.3)

    print(f"\n  Land cover records: {len(results)}")

    output = {
        "source": "EPA StreamCat NLCD2019 + MRLC fallback",
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "land_cover_data": results,
    }
    out = REAL_DATA_DIR / "urban_landcover_real.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ Saved to {out}")
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM 7: Seasonal Flow Rates
# Source: USGS NWIS (National Water Information System) daily values
#         Queried for the nearest stream gauge to each hotspot
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_seasonal_flows():
    """
    For each hotspot, find the nearest USGS stream gauge and fetch
    monthly average flow rates over the past 2 years to get seasonal patterns.
    """
    print("\n" + "="*70)
    print("ITEM 7: Seasonal Flow Rates (USGS NWIS)")
    print("="*70)

    results = []

    for hs in HOTSPOTS:
        # Step 1: Find nearest USGS gauges using site service
        # Parameter 00060 = Discharge (cubic feet per second)
        # Search within 30km bounding box
        dlat = 0.27  # ~30km
        dlng = 0.35
        site_url = (
            f"https://waterservices.usgs.gov/nwis/site/?"
            f"format=rdb&bBox={hs['lng']-dlng},{hs['lat']-dlat},{hs['lng']+dlng},{hs['lat']+dlat}"
            f"&parameterCd=00060&siteType=ST&siteStatus=active&hasDataTypeCd=dv"
        )

        # We need to parse RDB format - let's use JSON stat service instead
        stat_url = (
            f"https://waterservices.usgs.gov/nwis/site/?"
            f"format=mapper&bBox={hs['lng']-dlng},{hs['lat']-dlat},{hs['lng']+dlng},{hs['lat']+dlat}"
            f"&parameterCd=00060&siteType=ST&siteStatus=active&hasDataTypeCd=dv"
        )
        # Actually use the JSON output for site service
        site_url_json = (
            f"https://waterservices.usgs.gov/nwis/iv/?"
            f"format=json&bBox={hs['lng']-dlng},{hs['lat']-dlat},{hs['lng']+dlng},{hs['lat']+dlat}"
            f"&parameterCd=00060&siteType=ST&siteStatus=active&period=P1D"
        )
        site_data = fetch_json(site_url_json, f"NWIS gauges near {hs['id']}")

        nearest_site = None
        min_dist = float("inf")

        if site_data and "value" in site_data and "timeSeries" in site_data["value"]:
            for ts in site_data["value"]["timeSeries"]:
                si = ts.get("sourceInfo", {})
                geo = si.get("geoLocation", {}).get("geogLocation", {})
                slat = geo.get("latitude")
                slng = geo.get("longitude")
                site_code = si.get("siteCode", [{}])[0].get("value", "")
                site_name = si.get("siteName", "")

                if slat and slng:
                    d = haversine_km(hs["lat"], hs["lng"], slat, slng)
                    if d < min_dist:
                        min_dist = d
                        # Get the latest reading
                        vals = ts.get("values", [{}])[0].get("value", [])
                        latest_cfs = float(vals[-1]["value"]) if vals else None

                        nearest_site = {
                            "site_code": site_code,
                            "site_name": site_name,
                            "lat": slat,
                            "lng": slng,
                            "distance_km": round(d, 2),
                            "latest_discharge_cfs": latest_cfs,
                            "latest_discharge_m3s": round(latest_cfs * 0.028317, 4) if latest_cfs else None,
                        }

        # Step 2: Get monthly statistics for the nearest gauge
        monthly_flows = {}
        if nearest_site:
            # Fetch daily values for past 2 years for seasonal analysis
            stat2_url = (
                f"https://waterservices.usgs.gov/nwis/stat/?"
                f"format=json&sites={nearest_site['site_code']}"
                f"&parameterCd=00060&statReportType=monthly&statTypeCd=mean"
            )
            stat_data = fetch_json(stat2_url, f"NWIS monthly stats {nearest_site['site_code']}")

            if stat_data and "value" in stat_data and "timeSeries" in stat_data["value"]:
                for ts in stat_data["value"]["timeSeries"]:
                    for val_group in ts.get("values", []):
                        for v in val_group.get("value", []):
                            month_str = v.get("dateTime", "")
                            # Stat service returns dateTime like "2024-01" or just month number
                            try:
                                flow_cfs = float(v.get("value", 0))
                                # Extract month
                                if "-" in month_str:
                                    month = int(month_str.split("-")[1])
                                else:
                                    month = int(month_str)
                                month_name = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month]
                                if month_name not in monthly_flows or flow_cfs > 0:
                                    monthly_flows[month_name] = {
                                        "month": month,
                                        "mean_flow_cfs": round(flow_cfs, 1),
                                        "mean_flow_m3s": round(flow_cfs * 0.028317, 3),
                                    }
                            except (ValueError, IndexError):
                                continue

            # If monthly stats unavailable, try daily values for last year and compute ourselves
            if not monthly_flows:
                dv_url = (
                    f"https://waterservices.usgs.gov/nwis/dv/?"
                    f"format=json&sites={nearest_site['site_code']}"
                    f"&parameterCd=00060&period=P365D"
                )
                dv_data = fetch_json(dv_url, f"NWIS daily values {nearest_site['site_code']}")

                if dv_data and "value" in dv_data and "timeSeries" in dv_data["value"]:
                    monthly_totals = {}
                    monthly_counts = {}
                    for ts in dv_data["value"]["timeSeries"]:
                        for val_group in ts.get("values", []):
                            for v in val_group.get("value", []):
                                try:
                                    dt = v.get("dateTime", "")[:10]
                                    flow = float(v.get("value", 0))
                                    month = int(dt.split("-")[1])
                                    month_name = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month]
                                    monthly_totals[month_name] = monthly_totals.get(month_name, 0) + flow
                                    monthly_counts[month_name] = monthly_counts.get(month_name, 0) + 1
                                except (ValueError, IndexError):
                                    continue

                    for mn in monthly_totals:
                        avg_cfs = monthly_totals[mn] / monthly_counts[mn]
                        monthly_flows[mn] = {
                            "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"].index(mn) + 1,
                            "mean_flow_cfs": round(avg_cfs, 1),
                            "mean_flow_m3s": round(avg_cfs * 0.028317, 3),
                            "n_days": monthly_counts[mn],
                        }

        # Compute seasonal summary
        seasonal = {}
        if monthly_flows:
            flows_list = [(k, v["mean_flow_m3s"]) for k, v in monthly_flows.items()]
            if flows_list:
                max_month = max(flows_list, key=lambda x: x[1])
                min_month = min(flows_list, key=lambda x: x[1])
                seasonal = {
                    "peak_flow_month": max_month[0],
                    "peak_flow_m3s": max_month[1],
                    "low_flow_month": min_month[0],
                    "low_flow_m3s": min_month[1],
                    "seasonal_ratio": round(max_month[1] / min_month[1], 2) if min_month[1] > 0 else None,
                }

        results.append({
            "hotspot_id": hs["id"],
            "hotspot_name": hs["name"],
            "nearest_gauge": nearest_site,
            "monthly_mean_flows": monthly_flows if monthly_flows else None,
            "seasonal_summary": seasonal if seasonal else None,
        })
        time.sleep(0.5)

    gauges_found = sum(1 for r in results if r["nearest_gauge"])
    monthly_found = sum(1 for r in results if r["monthly_mean_flows"])
    print(f"\n  Gauges found: {gauges_found}/{len(results)}")
    print(f"  Monthly flows: {monthly_found}/{len(results)}")

    output = {
        "source": "USGS NWIS Water Services",
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "flow_data": results,
    }
    out = REAL_DATA_DIR / "seasonal_flows_real.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ Saved to {out}")
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM 8: Fish Species Presence by Watershed
# Source: GBIF (Global Biodiversity Information Facility)
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_fish_species_presence():
    """
    For each hotspot and each of the 8 target fish species, query GBIF
    for occurrence records within a geographic radius. This tells us
    which species are actually present in each watershed.
    """
    print("\n" + "="*70)
    print("ITEM 8: Fish Species Presence by Watershed (GBIF)")
    print("="*70)

    results = []
    search_radius_deg = 0.5  # ~55km at mid-latitudes

    for hs in HOTSPOTS:
        species_presence = []
        print(f"\n  Hotspot: {hs['id']}")

        for sp in FISH_SPECIES:
            # GBIF occurrence search within bounding box
            url = (
                f"https://api.gbif.org/v1/occurrence/search?"
                f"decimalLatitude={hs['lat']-search_radius_deg},{hs['lat']+search_radius_deg}"
                f"&decimalLongitude={hs['lng']-search_radius_deg},{hs['lng']+search_radius_deg}"
                f"&scientificName={urllib.parse.quote(sp['scientific'])}"
                f"&limit=5&hasCoordinate=true&hasGeospatialIssue=false"
                f"&basisOfRecord=HUMAN_OBSERVATION&basisOfRecord=OBSERVATION"
                f"&basisOfRecord=MACHINE_OBSERVATION&basisOfRecord=MATERIAL_SAMPLE"
            )
            gbif_data = fetch_json(url, f"GBIF {sp['common']} near {hs['id']}")

            count = 0
            nearest_record = None
            if gbif_data:
                count = gbif_data.get("count", 0)
                records = gbif_data.get("results", [])
                # Find nearest occurrence to hotspot
                min_d = float("inf")
                for rec in records:
                    rlat = rec.get("decimalLatitude")
                    rlng = rec.get("decimalLongitude")
                    if rlat and rlng:
                        d = haversine_km(hs["lat"], hs["lng"], rlat, rlng)
                        if d < min_d:
                            min_d = d
                            nearest_record = {
                                "lat": rlat,
                                "lng": rlng,
                                "distance_km": round(d, 2),
                                "year": rec.get("year"),
                                "dataset": rec.get("datasetName", "")[:60],
                                "basis": rec.get("basisOfRecord"),
                            }

            species_presence.append({
                "common_name": sp["common"],
                "scientific_name": sp["scientific"],
                "occurrence_count": count,
                "present": count > 0,
                "nearest_record": nearest_record,
            })
            time.sleep(0.35)  # GBIF rate limit

        present_count = sum(1 for s in species_presence if s["present"])
        results.append({
            "hotspot_id": hs["id"],
            "hotspot_name": hs["name"],
            "lat": hs["lat"],
            "lng": hs["lng"],
            "species_present_count": present_count,
            "total_species_checked": len(FISH_SPECIES),
            "species": species_presence,
        })

    # Summary stats
    total_presence = sum(r["species_present_count"] for r in results)
    total_checks = sum(r["total_species_checked"] for r in results)
    print(f"\n  Total species-location checks: {total_checks}")
    print(f"  Confirmed present: {total_presence} ({100*total_presence/total_checks:.1f}%)")

    output = {
        "source": "GBIF (Global Biodiversity Information Facility)",
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "fish_presence_data": results,
    }
    out = REAL_DATA_DIR / "fish_species_presence_real.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ Saved to {out}")
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION: Merge all data into a single enriched file
# ═══════════════════════════════════════════════════════════════════════════════
def integrate_all_data(facilities, streams, landcover, flows, fish):
    """Merge all 5 data sources into one enriched hotspot dataset."""
    print("\n" + "="*70)
    print("INTEGRATION: Merging all environmental data")
    print("="*70)

    enriched = []

    for hs in HOTSPOTS:
        record = {
            "hotspot_id": hs["id"],
            "hotspot_name": hs["name"],
            "lat": hs["lat"],
            "lng": hs["lng"],
            "state": hs["state"],
        }

        # Item 3: Facility metrics
        fac_match = next((f for f in facilities.get("hotspot_metrics", []) if f["hotspot_id"] == hs["id"]), None)
        if fac_match:
            record["pfas_facility_count_50km"] = fac_match["pfas_facility_count_50km"]
            record["nearest_pfas_facility_km"] = fac_match["nearest_pfas_facility_km"]
            record["nearby_facilities_top5"] = fac_match["nearby_facilities"][:5]
        else:
            record["pfas_facility_count_50km"] = None
            record["nearest_pfas_facility_km"] = None

        # Item 4: Stream attributes
        stream_match = next((s for s in streams.get("hotspot_stream_attributes", []) if s["hotspot_id"] == hs["id"]), None)
        if stream_match:
            record["comid"] = stream_match["comid"]
            record["stream_order"] = stream_match["stream_order"]
            record["watershed_area_km2"] = stream_match["watershed_area_km2"]
            record["reach_length_km"] = stream_match["reach_length_km"]
        else:
            record["comid"] = None
            record["stream_order"] = None
            record["watershed_area_km2"] = None

        # Item 5: Land cover
        if stream_match and stream_match.get("comid"):
            lc_match = next((l for l in landcover.get("land_cover_data", []) if str(l.get("comid")) == str(stream_match["comid"])), None)
            if lc_match:
                record["pct_urban"] = lc_match.get("pct_urban_watershed") or lc_match.get("pct_urban_catchment")
                record["pct_impervious"] = lc_match.get("pct_impervious_watershed") or lc_match.get("pct_impervious_catchment")
                record["pct_forest"] = lc_match.get("pct_forest_catchment")
                record["pct_agriculture"] = lc_match.get("pct_agriculture_catchment")
            else:
                record["pct_urban"] = None
                record["pct_impervious"] = None
        else:
            record["pct_urban"] = None
            record["pct_impervious"] = None

        # Item 7: Flow rates
        flow_match = next((f for f in flows.get("flow_data", []) if f["hotspot_id"] == hs["id"]), None)
        if flow_match:
            record["nearest_gauge"] = flow_match["nearest_gauge"]
            record["monthly_flows"] = flow_match["monthly_mean_flows"]
            record["seasonal_summary"] = flow_match["seasonal_summary"]
        else:
            record["nearest_gauge"] = None
            record["monthly_flows"] = None

        # Item 8: Fish species
        fish_match = next((f for f in fish.get("fish_presence_data", []) if f["hotspot_id"] == hs["id"]), None)
        if fish_match:
            record["species_present_count"] = fish_match["species_present_count"]
            record["species_presence"] = fish_match["species"]
        else:
            record["species_present_count"] = None
            record["species_presence"] = []

        enriched.append(record)

    # ── Summary statistics ──
    summary = {
        "total_hotspots": len(enriched),
        "facilities_with_data": sum(1 for r in enriched if r.get("pfas_facility_count_50km") is not None),
        "stream_order_available": sum(1 for r in enriched if r.get("stream_order") is not None),
        "watershed_area_available": sum(1 for r in enriched if r.get("watershed_area_km2") is not None),
        "land_cover_available": sum(1 for r in enriched if r.get("pct_urban") is not None),
        "flow_data_available": sum(1 for r in enriched if r.get("nearest_gauge") is not None),
        "fish_presence_available": sum(1 for r in enriched if r.get("species_present_count") is not None),
        "total_facilities_in_db": facilities.get("total_unique_facilities", 0),
    }

    output = {
        "source": "TrophicTrace Integrated Environmental Data",
        "version": "2.0",
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "data_sources": [
            "EPA ECHO (Enforcement & Compliance History Online)",
            "EPA TRI (Toxics Release Inventory)",
            "USGS NLDI (Network Linked Data Index)",
            "NHDPlus (National Hydrography Dataset Plus)",
            "EPA StreamCat (Stream Catchment Dataset)",
            "USGS NWIS (National Water Information System)",
            "GBIF (Global Biodiversity Information Facility)",
        ],
        "summary": summary,
        "hotspots": enriched,
    }

    out = REAL_DATA_DIR / "enriched_environmental_data.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ Integrated data saved to {out}")

    # Print summary
    print(f"\n  ── Data Completeness Summary ──")
    for k, v in summary.items():
        if isinstance(v, int) and "available" in k:
            pct = 100 * v / summary["total_hotspots"]
            print(f"    {k}: {v}/{summary['total_hotspots']} ({pct:.0f}%)")
        else:
            print(f"    {k}: {v}")

    return output


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  TrophicTrace Environmental Data Fetcher                       ║")
    print("║  Fetching real data for items 3, 4, 5, 7, 8                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Run each fetcher in sequence
    facilities = fetch_pfas_facilities()
    streams = fetch_stream_attributes()
    landcover = fetch_urban_landcover(streams)
    flows = fetch_seasonal_flows()
    fish = fetch_fish_species_presence()

    # Integrate all
    integrated = integrate_all_data(facilities, streams, landcover, flows, fish)

    print("\n" + "="*70)
    print("DONE! All real environmental data fetched and integrated.")
    print("="*70)
    print(f"\nOutput files in {REAL_DATA_DIR}:")
    for f in sorted(REAL_DATA_DIR.glob("*_real.json")):
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")
    print(f"  enriched_environmental_data.json ({(REAL_DATA_DIR / 'enriched_environmental_data.json').stat().st_size:,} bytes)")
