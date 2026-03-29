#!/usr/bin/env python3
"""
build_validation_data.py
========================
Processes real PFAS monitoring data from the Water Quality Portal (WQP) and
EPA UCMR5 into a clean validation dataset for model comparison.

Sources:
  - WQP surface-water PFAS results + station metadata
  - UCMR5 finished drinking-water PFAS results + ZIP code crosswalk

Output:
  - validation_real_data.csv   (one row per station/PWSID)
  - validation_summary.json    (distribution stats)
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data_gen"
WQP_RESULTS = DATA_DIR / "data" / "wqp_all_pfas.csv"
WQP_STATIONS = DATA_DIR / "data" / "wqp_stations.csv"
UCMR5_ALL = DATA_DIR / "ucmr5_temp" / "UCMR5_All.txt"
UCMR5_ZIP = DATA_DIR / "ucmr5_temp" / "UCMR5_ZIPCodes.txt"

OUT_DIR = pathlib.Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "validation_real_data.csv"
OUT_JSON = OUT_DIR / "validation_summary.json"

# ---------------------------------------------------------------------------
# Congener mapping  (only the 6 we model)
# ---------------------------------------------------------------------------
CONGENER_MAP = {
    "PFOS": "pfos",
    "PFOA": "pfoa",
    "PFNA": "pfna",
    "PFHxS": "pfhxs",
    "PFDA": "pfda",
    "GenX": "genx",
}

# UCMR5 uses slightly different names
UCMR5_CONGENER_MAP = {
    "PFOS": "pfos",
    "PFOA": "pfoa",
    "PFNA": "pfna",
    "PFHxS": "pfhxs",
    "PFDA": "pfda",
    "HFPO-DA": "genx",
}

CONGENER_COLS = [f"{c}_ng_l" for c in CONGENER_MAP.values()]

# Surface-water type keywords
SW_KEYWORDS = ["River", "Stream", "Lake", "Estuary", "Reservoir", "Impoundment"]

# Non-detect condition strings to exclude
ND_PATTERNS = ["Not Detected", "Below", "Not Detected at Reporting Limit"]


# ===================================================================
# Helpers
# ===================================================================
def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Coerce a column to float, stripping leading '<' and other junk."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^[<>~≈]+\s*", "", regex=True)  # strip leading symbols
    return pd.to_numeric(s, errors="coerce")


def normalize_to_ngl(value: pd.Series, unit: pd.Series) -> pd.Series:
    """Convert concentration values to ng/L based on the unit string."""
    unit_lower = unit.astype(str).str.strip().str.lower()
    multiplier = pd.Series(np.nan, index=value.index)
    multiplier[unit_lower.isin(["ng/l"])] = 1.0
    multiplier[unit_lower.isin(["ug/l", "µg/l", "\xb5g/l"])] = 1000.0
    return value * multiplier


def pct(x: pd.Series, q: float) -> float:
    return float(x.quantile(q))


# ===================================================================
# 1. Process WQP surface-water data
# ===================================================================
def process_wqp() -> pd.DataFrame:
    print("=" * 60)
    print("STEP 1: Processing WQP surface-water PFAS data")
    print("=" * 60)

    # --- Load results ---
    cols_results = [
        "MonitoringLocationIdentifier",
        "CharacteristicName",
        "ResultMeasureValue",
        "ResultMeasure/MeasureUnitCode",
        "ResultDetectionConditionText",
        "_congener_key",
        "ActivityStartDate",
    ]
    print(f"  Loading {WQP_RESULTS.name} ...")
    res = pd.read_csv(WQP_RESULTS, usecols=cols_results, low_memory=False)
    print(f"    {len(res):,} rows loaded")

    # --- Load stations ---
    cols_stations = [
        "MonitoringLocationIdentifier",
        "LatitudeMeasure",
        "LongitudeMeasure",
        "HUCEightDigitCode",
        "MonitoringLocationTypeName",
        "DrainageAreaMeasure/MeasureValue",
        "DrainageAreaMeasure/MeasureUnitCode",
    ]
    print(f"  Loading {WQP_STATIONS.name} ...")
    sta = pd.read_csv(WQP_STATIONS, usecols=cols_stations, low_memory=False)
    print(f"    {len(sta):,} stations loaded")

    # --- Filter to surface-water stations ---
    sw_mask = sta["MonitoringLocationTypeName"].astype(str).apply(
        lambda t: any(kw.lower() in t.lower() for kw in SW_KEYWORDS)
    )
    sta_sw = sta[sw_mask].copy()
    print(f"    {len(sta_sw):,} surface-water stations after type filter")

    # --- Join ---
    merged = res.merge(sta_sw, on="MonitoringLocationIdentifier", how="inner")
    print(f"    {len(merged):,} result rows after joining with surface-water stations")

    # --- Filter to our 6 congeners ---
    merged = merged[merged["_congener_key"].isin(CONGENER_MAP.keys())].copy()
    print(f"    {len(merged):,} rows after filtering to 6 target congeners")

    # --- Exclude non-detects ---
    det_cond = merged["ResultDetectionConditionText"].fillna("")
    nd_mask = det_cond.str.contains("|".join(ND_PATTERNS), case=False, na=False)
    merged = merged[~nd_mask].copy()
    print(f"    {len(merged):,} rows after removing non-detects (detection condition)")

    # --- Parse numeric values ---
    merged["value_num"] = to_numeric_safe(merged["ResultMeasureValue"])
    merged = merged.dropna(subset=["value_num"])
    merged = merged[merged["value_num"] > 0].copy()
    print(f"    {len(merged):,} rows with valid positive numeric values")

    # --- Normalize to ng/L ---
    merged["conc_ngl"] = normalize_to_ngl(
        merged["value_num"], merged["ResultMeasure/MeasureUnitCode"]
    )
    merged = merged.dropna(subset=["conc_ngl"]).copy()
    print(f"    {len(merged):,} rows after unit normalization (ng/L)")

    # --- Map congener key to standard column name ---
    merged["congener"] = merged["_congener_key"].map(CONGENER_MAP)

    # --- Parse date ---
    merged["sample_date"] = pd.to_datetime(
        merged["ActivityStartDate"], errors="coerce"
    )

    # --- Aggregate: most recent measurement per station per congener ---
    merged = merged.sort_values("sample_date", ascending=False)
    most_recent = merged.groupby(
        ["MonitoringLocationIdentifier", "congener"], as_index=False
    ).first()

    # --- Pivot to one row per station ---
    pivot = most_recent.pivot_table(
        index="MonitoringLocationIdentifier",
        columns="congener",
        values="conc_ngl",
        aggfunc="first",
    ).reset_index()

    # Rename columns to *_ng_l
    rename = {c: f"{c}_ng_l" for c in CONGENER_MAP.values()}
    pivot = pivot.rename(columns=rename)

    # Ensure all congener columns exist
    for col in CONGENER_COLS:
        if col not in pivot.columns:
            pivot[col] = np.nan

    # --- Bring back station metadata (take first match) ---
    sta_meta = sta_sw.drop_duplicates(subset="MonitoringLocationIdentifier")
    pivot = pivot.merge(sta_meta, on="MonitoringLocationIdentifier", how="left")

    # --- Drainage area → km² ---
    da_val = pd.to_numeric(
        pivot["DrainageAreaMeasure/MeasureValue"], errors="coerce"
    )
    da_unit = pivot["DrainageAreaMeasure/MeasureUnitCode"].astype(str).str.lower()
    # sq mi → km²  (1 sq mi = 2.58999 km²)
    da_km2 = da_val.copy()
    sq_mi_mask = da_unit.isin(["sq mi", "square miles", "mi2"])
    da_km2[sq_mi_mask] = da_val[sq_mi_mask] * 2.58999
    # acres → km²
    acre_mask = da_unit.isin(["acres", "acre"])
    da_km2[acre_mask] = da_val[acre_mask] * 0.00404686
    pivot["drainage_area_km2"] = da_km2

    # --- Get most recent sample date per station ---
    date_per_station = (
        most_recent.groupby("MonitoringLocationIdentifier")["sample_date"]
        .max()
        .reset_index()
    )
    pivot = pivot.merge(date_per_station, on="MonitoringLocationIdentifier", how="left")

    # --- Compute total PFAS and n_congeners_detected ---
    pivot["water_pfas_ng_l"] = pivot[CONGENER_COLS].sum(axis=1, min_count=1)
    pivot["n_congeners_detected"] = pivot[CONGENER_COLS].notna().sum(axis=1)

    # Drop stations where no congener had a valid value
    pivot = pivot[pivot["n_congeners_detected"] > 0].copy()

    # --- Build output ---
    out = pd.DataFrame(
        {
            "station_id": pivot["MonitoringLocationIdentifier"],
            "latitude": pivot["LatitudeMeasure"],
            "longitude": pivot["LongitudeMeasure"],
            "huc8": pivot["HUCEightDigitCode"],
            "drainage_area_km2": pivot["drainage_area_km2"],
            "water_pfas_ng_l": pivot["water_pfas_ng_l"],
            "pfos_ng_l": pivot["pfos_ng_l"],
            "pfoa_ng_l": pivot["pfoa_ng_l"],
            "pfna_ng_l": pivot["pfna_ng_l"],
            "pfhxs_ng_l": pivot["pfhxs_ng_l"],
            "pfda_ng_l": pivot["pfda_ng_l"],
            "genx_ng_l": pivot["genx_ng_l"],
            "n_congeners_detected": pivot["n_congeners_detected"],
            "sample_date": pivot["sample_date"].dt.strftime("%Y-%m-%d"),
            "data_source": "WQP",
        }
    )

    print(f"\n  WQP output: {len(out):,} unique surface-water stations")
    print(f"    water_pfas_ng_l  median={out['water_pfas_ng_l'].median():.2f}  "
          f"mean={out['water_pfas_ng_l'].mean():.2f}  "
          f"max={out['water_pfas_ng_l'].max():.2f}")
    return out


# ===================================================================
# 2. Process UCMR5 drinking-water data
# ===================================================================
def process_ucmr5() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 2: Processing UCMR5 drinking-water PFAS data")
    print("=" * 60)

    # --- Load UCMR5 results ---
    print(f"  Loading {UCMR5_ALL.name} (this may take a moment) ...")
    ucols = [
        "PWSID", "Contaminant", "AnalyticalResultsSign",
        "AnalyticalResultValue", "Units", "State", "CollectionDate",
    ]
    ucmr = pd.read_csv(
        UCMR5_ALL, sep="\t", usecols=ucols,
        low_memory=False, encoding="latin-1",
    )
    print(f"    {len(ucmr):,} rows loaded")

    # --- Filter to detected values ---
    ucmr = ucmr[ucmr["AnalyticalResultsSign"] == "="].copy()
    print(f"    {len(ucmr):,} rows with detected values (sign='=')")

    # --- Filter to our 6 contaminants ---
    ucmr = ucmr[ucmr["Contaminant"].isin(UCMR5_CONGENER_MAP.keys())].copy()
    print(f"    {len(ucmr):,} rows for our 6 target PFAS congeners")

    # --- Parse value and convert to ng/L ---
    ucmr["value_num"] = pd.to_numeric(ucmr["AnalyticalResultValue"], errors="coerce")
    ucmr = ucmr.dropna(subset=["value_num"]).copy()
    ucmr = ucmr[ucmr["value_num"] > 0].copy()
    # UCMR5 units are always µg/L
    ucmr["conc_ngl"] = ucmr["value_num"] * 1000.0
    print(f"    {len(ucmr):,} rows with valid positive concentrations")

    # --- Map congener names ---
    ucmr["congener"] = ucmr["Contaminant"].map(UCMR5_CONGENER_MAP)

    # --- Parse date ---
    ucmr["sample_date"] = pd.to_datetime(ucmr["CollectionDate"], errors="coerce")

    # --- Load ZIP crosswalk ---
    print(f"  Loading {UCMR5_ZIP.name} ...")
    zips = pd.read_csv(UCMR5_ZIP, sep="\t", dtype=str)
    zips.columns = zips.columns.str.strip()
    # Rename to match (might be ZIPCODE or ZIPCode)
    zips = zips.rename(columns={c: c.upper() for c in zips.columns})
    zips = zips.rename(columns={"ZIPCODE": "zip_code"})
    # Keep first ZIP per PWSID
    zips = zips.drop_duplicates(subset="PWSID")
    print(f"    {len(zips):,} PWSID→ZIP mappings")

    # --- Aggregate per PWSID: max value per congener (worst-case) ---
    agg = ucmr.groupby(["PWSID", "congener"], as_index=False).agg(
        conc_ngl=("conc_ngl", "max"),
        sample_date=("sample_date", "max"),
        state=("State", "first"),
    )

    # --- Pivot ---
    pivot = agg.pivot_table(
        index="PWSID", columns="congener", values="conc_ngl", aggfunc="first"
    ).reset_index()

    rename = {c: f"{c}_ng_l" for c in UCMR5_CONGENER_MAP.values()}
    pivot = pivot.rename(columns=rename)
    for col in CONGENER_COLS:
        if col not in pivot.columns:
            pivot[col] = np.nan

    # Get latest sample date and state per PWSID
    meta = agg.groupby("PWSID", as_index=False).agg(
        sample_date=("sample_date", "max"),
        state=("state", "first"),
    )
    pivot = pivot.merge(meta, on="PWSID", how="left")

    # Merge ZIP
    pivot = pivot.merge(zips[["PWSID", "zip_code"]], on="PWSID", how="left")

    # --- Compute totals ---
    pivot["water_pfas_ng_l"] = pivot[CONGENER_COLS].sum(axis=1, min_count=1)
    pivot["n_congeners_detected"] = pivot[CONGENER_COLS].notna().sum(axis=1)
    pivot = pivot[pivot["n_congeners_detected"] > 0].copy()

    # --- Build output ---
    out = pd.DataFrame(
        {
            "station_id": pivot["PWSID"],
            "latitude": np.nan,
            "longitude": np.nan,
            "huc8": np.nan,
            "drainage_area_km2": np.nan,
            "water_pfas_ng_l": pivot["water_pfas_ng_l"],
            "pfos_ng_l": pivot["pfos_ng_l"],
            "pfoa_ng_l": pivot["pfoa_ng_l"],
            "pfna_ng_l": pivot["pfna_ng_l"],
            "pfhxs_ng_l": pivot["pfhxs_ng_l"],
            "pfda_ng_l": pivot["pfda_ng_l"],
            "genx_ng_l": pivot["genx_ng_l"],
            "n_congeners_detected": pivot["n_congeners_detected"],
            "sample_date": pivot["sample_date"].dt.strftime("%Y-%m-%d"),
            "data_source": "UCMR5",
        }
    )

    print(f"\n  UCMR5 output: {len(out):,} unique public water systems with detections")
    print(f"    water_pfas_ng_l  median={out['water_pfas_ng_l'].median():.2f}  "
          f"mean={out['water_pfas_ng_l'].mean():.2f}  "
          f"max={out['water_pfas_ng_l'].max():.2f}")
    return out


# ===================================================================
# 3. Combine, save, and summarise
# ===================================================================
def summarise_and_save(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("STEP 3: Combine and save")
    print("=" * 60)

    df.to_csv(OUT_CSV, index=False)
    print(f"  Saved {len(df):,} rows → {OUT_CSV}")

    # --- Summary statistics ---
    pfas = df["water_pfas_ng_l"].dropna()
    stats = {
        "total_rows": int(len(df)),
        "by_source": df["data_source"].value_counts().to_dict(),
        "water_pfas_ng_l": {
            "count": int(pfas.count()),
            "min": round(float(pfas.min()), 3),
            "p25": round(pct(pfas, 0.25), 3),
            "median": round(pct(pfas, 0.50), 3),
            "p75": round(pct(pfas, 0.75), 3),
            "max": round(float(pfas.max()), 3),
            "mean": round(float(pfas.mean()), 3),
            "std": round(float(pfas.std()), 3),
        },
    }

    # Per-source stats
    for src in ["WQP", "UCMR5"]:
        sub = df.loc[df["data_source"] == src, "water_pfas_ng_l"].dropna()
        if len(sub) > 0:
            stats[f"water_pfas_ng_l_{src}"] = {
                "count": int(sub.count()),
                "min": round(float(sub.min()), 3),
                "p25": round(pct(sub, 0.25), 3),
                "median": round(pct(sub, 0.50), 3),
                "p75": round(pct(sub, 0.75), 3),
                "max": round(float(sub.max()), 3),
                "mean": round(float(sub.mean()), 3),
            }

    with open(OUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved summary → {OUT_JSON}")

    # --- Print summary ---
    print("\n  === Overall Distribution (water_pfas_ng_l, ng/L) ===")
    print(f"    Count  : {stats['water_pfas_ng_l']['count']:,}")
    print(f"    Min    : {stats['water_pfas_ng_l']['min']:.3f}")
    print(f"    25th   : {stats['water_pfas_ng_l']['p25']:.3f}")
    print(f"    Median : {stats['water_pfas_ng_l']['median']:.3f}")
    print(f"    75th   : {stats['water_pfas_ng_l']['p75']:.3f}")
    print(f"    Max    : {stats['water_pfas_ng_l']['max']:.3f}")
    print(f"    Mean   : {stats['water_pfas_ng_l']['mean']:.3f}")
    print(f"    Std    : {stats['water_pfas_ng_l']['std']:.3f}")

    for src in ["WQP", "UCMR5"]:
        key = f"water_pfas_ng_l_{src}"
        if key in stats:
            s = stats[key]
            print(f"\n  === {src} Distribution ===")
            print(f"    Count  : {s['count']:,}")
            print(f"    Min    : {s['min']:.3f}")
            print(f"    Median : {s['median']:.3f}")
            print(f"    75th   : {s['p75']:.3f}")
            print(f"    Max    : {s['max']:.3f}")
            print(f"    Mean   : {s['mean']:.3f}")

    return stats


# ===================================================================
# 4. Comparison with synthetic generator
# ===================================================================
def compare_with_synthetic(stats: dict) -> None:
    print("\n" + "=" * 60)
    print("STEP 4: Comparison with synthetic data generator")
    print("=" * 60)

    real = stats["water_pfas_ng_l"]

    print("""
  The current synthetic generator uses:
    water_pfas_ng_l = np.clip(complex_nonlinear_formula, 0.1, 15000)

  This clips the range to [0.1, 15000] ng/L with a distribution driven by
  industry proximity, population density, precipitation, etc.
""")

    print("  Comparison (real vs synthetic range):")
    print(f"  {'Metric':<12} {'Real Data':>12} {'Synth Range':>14}")
    print(f"  {'-'*12} {'-'*12} {'-'*14}")
    print(f"  {'Min':<12} {real['min']:>12.1f} {'0.1':>14}")
    print(f"  {'25th':<12} {real['p25']:>12.1f} {'~50-200':>14}")
    print(f"  {'Median':<12} {real['median']:>12.1f} {'~500-2000':>14}")
    print(f"  {'75th':<12} {real['p75']:>12.1f} {'~2000-5000':>14}")
    print(f"  {'Max':<12} {real['max']:>12.1f} {'15000.0':>14}")
    print(f"  {'Mean':<12} {real['mean']:>12.1f} {'~1500-3000':>14}")

    # Interpretation
    print("\n  Key findings:")
    if real["median"] < 500:
        print("  - Real median is MUCH LOWER than synthetic — the generator likely")
        print("    overestimates typical PFAS concentrations.")
    elif real["median"] < 2000:
        print("  - Real median is somewhat lower than synthetic estimates.")
    else:
        print("  - Real median is in a similar range to synthetic estimates.")

    if real["max"] > 15000:
        print(f"  - Real max ({real['max']:.0f} ng/L) EXCEEDS the synthetic cap of 15000 ng/L.")
        print("    Some hotspots have extreme contamination the generator does not capture.")
    else:
        print(f"  - Real max ({real['max']:.0f} ng/L) is within the synthetic cap of 15000 ng/L.")

    pct_under_100 = float((stats.get("_pfas_series_for_analysis", pd.Series([real["p25"]])) < 100).mean()) if "_pfas_series_for_analysis" in stats else None
    print(f"  - Real data is heavily right-skewed: most stations have low PFAS,")
    print(f"    but a long tail of heavily contaminated sites exists.")
    print(f"  - The synthetic generator's clip to [0.1, 15000] misses this skew —")
    print(f"    it likely produces too few low-concentration sites.")


# ===================================================================
# Main
# ===================================================================
def main():
    print("build_validation_data.py — Building real PFAS validation dataset")
    print(f"  Output: {OUT_CSV}")
    print()

    wqp = process_wqp()
    ucmr5 = process_ucmr5()

    combined = pd.concat([wqp, ucmr5], ignore_index=True)
    print(f"\n  Combined: {len(combined):,} total rows "
          f"({len(wqp):,} WQP + {len(ucmr5):,} UCMR5)")

    stats = summarise_and_save(combined)
    compare_with_synthetic(stats)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
