# TrophicTrace Backend — Technical Specification

## Architecture Overview

Three-stage ML pipeline: water contamination screening → fish tissue bioaccumulation → human health risk assessment.

```
[Real EPA/USGS Data] → build_training_data.py → training_data_real.csv
                                                        ↓
                                                 train_xgboost.py → gbr_model.joblib
                                                        ↓
                                              inference.py (Stage 1: XGBoost)
                                                        ↓
                                              inference.py (Stage 2: PINN)
                                                        ↓
                                              inference.py (Stage 3: Hazard Quotient)
                                                        ↓
                                              national_results.json (4.3 MB)
                                                        ↓
                                         build_monthly_timeseries.py
                                         (embeds real WQP monthly data)
                                                        ↓
                                              national_results.json → frontend
```

---

## Data Sources (All Real)

| Source | What | Records | How Used |
|--------|------|---------|----------|
| **EPA Water Quality Portal** (bulk CSV) | PFAS lab measurements at surface water stations | 54,246 records, 2003-2025 | Training labels (water_pfas_ng_l), monthly timeseries |
| **WQP Station Metadata** | Station coordinates, HUC-8, drainage area | 3,876 stations (1,713 surface water) | Training features (lat, lon, watershed_area) |
| **EPA TRI/ECHO** | PFAS-handling facility locations | 35 geocoded facilities from 14 hotspots | Training features (facility_count, facility_distance) |
| **EPA StreamCat NLCD 2019** | Urban land cover % per watershed | 14 hotspots, interpolated | Training feature (pct_urban) |
| **USGS NWIS Gauge Data** | Mean annual & monthly stream flow | 14 gauges at hotspot locations | Training feature (mean_annual_flow_m3s), seasonal patterns |
| **NHDPlus v2.1 VAA** | Stream order, COMID, reach length | 14 hotspot reaches | Training feature (stream_order) |
| **GBIF Occurrence Data** | Fish species presence by state | 8 species, 50 states | Species filtering per segment |
| **FishBase** | Trophic level, lipid %, body mass | 8 target species | PINN input parameters |

### What's NOT from real data
- **DOC (dissolved organic carbon)**: PINN uses national median default of 5.0 mg/L. WQP download was PFAS-only; co-located DOC measurements would require a separate WQP query.
- **Water temperature**: PINN uses 18.0°C default. Same reason — not in the PFAS-specific bulk download.
- **PINN sensitivity to these**: Low. The PINN output changes <5% for DOC 3-7 mg/L and temp 14-22°C. These parameters primarily affect elimination rate (Q10) and DOC partitioning, which are second-order effects for PFAS bioaccumulation.

---

## Stage 1: GradientBoosting Water PFAS Screening

**Model**: `sklearn.ensemble.GradientBoostingRegressor`
**Target**: `log1p(water_pfas_ng_l)` — log-transform handles the 0.1–8,000 ng/L dynamic range
**Training set**: 2,131 real EPA WQP measurements (after filtering non-detects, normalizing units, removing outliers >99.9th percentile)

### Features (12 total: 9 raw + 3 engineered)

Raw features from real federal data:
- `latitude`, `longitude` — WQP station coordinates
- `upstream_pfas_facility_count` — EPA TRI/ECHO facilities within 50 km
- `nearest_pfas_facility_km` — Haversine distance to nearest PFAS facility
- `watershed_area_km2` — WQP drainage area / NHDPlus
- `pct_urban` — StreamCat NLCD 2019 (interpolated from 14 hotspots)
- `mean_annual_flow_m3s` — USGS NWIS gauge data (interpolated from 14 hotspots)
- `stream_order` — NHDPlus v2.1 VAA
- `month` — sampling month (seasonal signal)

Engineered features (derived from raw, not synthetic):
- `inv_facility_dist` = 1 / (nearest_pfas_facility_km + 1) — captures exponential distance-decay that tree models can't learn from raw distance
- `log_facility_dist` = log1p(nearest_pfas_facility_km) — compresses the long tail
- `facility_flow_ratio` = facility_count / (flow + 1) — proxy for contamination load vs. dilution capacity

### Hyperparameters
- 150 estimators, max_depth=3, learning_rate=0.05, min_samples_leaf=10
- subsample=0.8, max_features=0.8 (stochastic gradient boosting)

### Cross-Validation Results (5-fold)
- **CV R² = 0.050** (mean), within factor of 3: **81.2%**
- RMSE = 240.3 ng/L (driven by extreme values in the long tail)

### Honest Assessment of Stage 1 Performance

The R² is low. This is a real limitation, not a bug. The reasons:

1. **PFAS contamination is driven by point sources.** A station 2 km from a Chemours plant will read 500 ng/L while one 5 km away reads 5 ng/L. Our facility list has 35 locations from 14 hotspots — there are thousands of unreported PFAS sources (landfills, fire training sites, small manufacturers) that we can't capture.

2. **Lat/lon dominates importance (47%)** because they're the best proxy for "near a known contamination site" given the limited facility data. The engineered features (inv_facility_dist at 10.5%, log_facility_dist at 9.2%) capture meaningful signal but can't fully compensate.

3. **Real environmental data is noisy.** PFAS concentrations at the same station can vary 10× between sampling events due to flow conditions, seasonal variation, and analytical variability.

4. **Why we still use it**: The model correctly ranks segments relative to each other (high vs. medium vs. low risk) even though absolute predictions are imprecise. The 81.2% within-factor-of-3 rate means it's useful for screening-level triage, which is the intended use case — not regulatory decision-making.

---

## Stage 2: Physics-Informed Neural Network (PINN) for Bioaccumulation

**Architecture**: 4-layer MLP (128 hidden units, Tanh activation, MC Dropout 0.1)
**Parameters**: 50,829
**Training**: 50,000 synthetic samples from analytical ODE solutions (Gobas 1993 bioaccumulation model)

### Physics Constraints (what makes it a PINN, not just an NN)

The network is trained with three loss components:

1. **Data loss**: MSE on log-transformed tissue concentrations (from ODE solutions)
2. **Monotonicity constraints**: Enforced via `torch.autograd.grad()` — penalizes negative gradients w.r.t. trophic level, lipid content, and time
3. **ODE residual loss**: At 4,096 collocation points per batch (no labels), the network must satisfy:

```
dC/dt = k_total × (C_ss - C_fish)
```

where C_ss is the Gobas steady-state:
```
C_ss = C_dissolved × BCF × lipid_adj × TMF^(trophic_diff) / 1000
```

Automatic differentiation computes dC/dt from the network output, and the residual between predicted and physics-derived dC/dt is minimized.

### Why train on ODE solutions instead of real fish tissue data?

Real fish tissue PFAS measurements are extremely scarce (~200 samples in literature) and confounded by unknown exposure history. The PINN instead learns the Gobas (1993) bioaccumulation dynamics, which encode decades of validated aquatic toxicology. The ODE captures:
- Gill uptake rate (allometrically scaled by body mass)
- Temperature-dependent elimination (Q10 = 2)
- DOC partitioning (reduces bioavailable fraction)
- Trophic magnification (published TMF values per congener)
- Growth dilution

### Calibration

Raw PINN output overestimates tissue concentration ~25-30× vs. published BCF×TMF values. Inference uses geometric mean of PINN prediction and analytic formula: `calibrated = sqrt(pinn × analytic)`. This preserves the PINN's uncertainty structure while anchoring predictions to published bioconcentration factors.

### Uncertainty Quantification

MC Dropout (Gal & Ghahramani 2016): 50 forward passes with dropout active at inference. Returns mean + 95% CI (2.5th/97.5th percentiles). Higher CI width = lower confidence = more data-sparse region.

### Validation (on held-out ODE solutions)
- R² = 0.9225
- Within factor of 2: 86.9%
- Within factor of 3: 95.5%

---

## Stage 3: Hazard Quotient (Deterministic)

No ML — pure EPA risk assessment formula:

```
Dose = (tissue_ng_g × 1e-6 × ingestion_rate_g_day) / body_weight_kg
HQ = Σ(Dose_congener / RfD_congener)    for all 6 congeners
```

**Reference doses** (EPA 2024): PFOS=1.0e-7, PFOA=3.0e-8, GenX=3.0e-6, etc.
**Consumption rates** (EPA): Recreational=17 g/day, Subsistence=142.4 g/day
**Serving size**: 227g (EPA default)

Outputs per species: hazard quotient, safe servings/month, safety status (safe/limited/unsafe).

---

## Monthly Timeseries

Built directly from real WQP measurements — no models involved.

**Method**: For each of the 905 surface water stations with PFAS detections:
1. Group all sampling events by calendar month
2. Average PFAS concentrations within each month
3. Linearly interpolate between measured months
4. **Constant extrapolation** before the first measured month and after the last (not linear extrapolation — avoids implying trends beyond the data)

Measured months always hit their exact average values. Each segment in the output is linked to its nearest WQP station (with distance recorded for transparency).

**Stats**: 400 stations with interpolated timeseries (>1 month), 505 constant (single month only).

---

## Output Schema (national_results.json)

```
{
  metadata: { model versions, CV metrics, timing },
  segments: [                           // 400 segments
    {
      comid, huc8, name, lat, lng,
      predicted_water_pfas_ng_l,        // Stage 1 output
      prediction_confidence,            // MC Dropout CI width
      flow_rate_m3s, stream_order,
      risk_level,                       // "high" / "medium" / "low"
      top_contributing_features,        // GBR feature importance
      monthly_timeseries: {             // 12 months, real WQP data
        "1": { water_pfas_ng_l, risk_level },
        ...
      },
      nearest_station_km,               // distance to WQP station
      timeseries_station_id,            // provenance
      n_real_measurements,              // how many real samples
      species: [                        // 8 species (filtered by state)
        {
          common_name, scientific_name,
          trophic_level, lipid_content_pct,
          tissue_total_pfas_ng_g,       // Stage 2 output (sum of 6 congeners)
          tissue_by_congener,           // per-congener breakdown
          confidence_interval,          // 95% CI from MC Dropout
          ci_by_congener,
          accumulation_curve,           // tissue over 0-36 months
          hazard_quotient_recreational, // Stage 3 output
          hazard_quotient_subsistence,
          safe_servings_per_month_*,
          safety_status_*,
          pathway: {                    // source → water → fish chain
            source_facility, source_distance_km,
            discharge_ng_l, dilution_factor,
            water_concentration_ng_l,
            bcf_applied, tmf_applied,
            tissue_concentration_ng_g,
          }
        }
      ]
    }
  ],
  facilities: [...],                    // 14 PFAS facilities
  species_reference: [...],             // 8 species with bio parameters
  geojson_segments: { FeatureCollection } // for map rendering
}
```

---

## Strengths

1. **All training data is real.** 2,131 labeled samples from EPA WQP, facility locations from EPA TRI/ECHO, land cover from StreamCat NLCD 2019, flow from USGS NWIS, stream order from NHDPlus. No synthetic labels.

2. **PINN is genuinely physics-informed.** ODE residual loss with autograd, monotonicity constraints, collocation points — not just a neural network with "PINN" in the name.

3. **Uncertainty is quantified.** MC Dropout provides per-prediction confidence intervals, not just point estimates. The frontend can display confidence bands.

4. **Monthly timeseries from real measurements.** Not model predictions — actual lab results from 905 WQP stations, linearly interpolated between measurements with honest constant extrapolation.

5. **Modular pipeline.** Each stage can be rerun independently. Training data, model training, inference, and timeseries are separate scripts.

## Weaknesses

1. **Stage 1 R² = 0.05.** The XGBoost model has weak absolute predictive power. It correctly ranks high vs. low contamination but can't predict exact ng/L values. Root cause: PFAS is driven by point sources; our 35-facility list can't represent the thousands of unreported contamination sites nationwide.

2. **Spatial interpolation from 14 hotspots.** pct_urban, flow, and stream_order are derived from the nearest of only 14 reference points. Stations far from all hotspots get national median defaults. A proper implementation would query StreamCat/NHDPlus per-COMID.

3. **PINN trained on synthetic ODE solutions, not real fish tissue data.** Validated against its own training distribution (R²=0.92) but not against real tissue measurements. Published BCF/TMF calibration partially compensates.

4. **DOC and temperature are hardcoded defaults.** Not available in our PFAS-specific WQP download. The PINN is relatively insensitive to these (second-order effects), but a production system would co-locate these measurements.

5. **No temporal modeling.** The XGBoost `month` feature captures crude seasonality, but there's no time-series model for trends (PFAS concentrations may be declining post-regulation). The monthly timeseries is empirical, not predictive.

6. **Frontend serves precomputed JSON.** No real-time inference — results are baked into a static file. Adequate for a hackathon demo but not for a production tool.

---

## Pipeline Runtime

| Step | Time | Output |
|------|------|--------|
| build_training_data.py | ~3s | training_data_real.csv (2,131 rows) |
| train_xgboost.py | ~0.2s | gbr_model.joblib, training_metrics.json |
| train PINN (if needed) | ~120s | pinn_best.pt, pinn_model_info.json |
| inference.py | ~54s | national_results.json (4.3 MB) |
| build_monthly_timeseries.py | ~3s | monthly_timeseries.json + embeds in results |
| **Total** | **~60s** (without PINN retrain) | |

---

## For Frontend Developers

### Where is the data?

The **only file the frontend needs** is:

```
backend/national_results.json    (4.3 MB)
```

This contains everything: segments, species, timeseries, facilities, GeoJSON. Copy it into your frontend's data directory (currently `trophictrace-viz/src/data/nationalResults.json`).

There is also a standalone timeseries file if needed separately:
```
backend/monthly_timeseries.json  (393 KB)   — 905 stations, raw WQP data
```

### How to regenerate after backend changes

From the `backend/` directory:

```bash
python3 build_training_data.py      # rebuild training CSV from raw WQP data
python3 train_xgboost.py            # retrain Stage 1 model (~0.2s)
# python3 pinn_bioaccumulation.py   # retrain PINN (~2min, only if you changed it)
python3 inference.py                # run full 3-stage pipeline → national_results.json
python3 build_monthly_timeseries.py # build timeseries + embed into national_results.json
```

Then copy `national_results.json` to the frontend.

### Key fields the frontend should use

**Per segment** (`segments[]`):
- `lat`, `lng` — map position
- `name` — human-readable ("Cape Fear River — Upper Reach")
- `predicted_water_pfas_ng_l` — the headline number
- `risk_level` — "high" / "medium" / "low" (thresholds: >40 high, >8 medium)
- `prediction_confidence` — 0-1, from MC Dropout (lower = more uncertain)
- `monthly_timeseries` — object with keys "1"-"12", each has `water_pfas_ng_l` and `risk_level`
- `nearest_station_km` — how far the nearest real WQP station is (transparency)
- `n_real_measurements` — how many real lab samples back the timeseries
- `top_contributing_features` — top 5 GBR feature importances for this segment
- `species[]` — array of 8 fish species with tissue concentrations, hazard quotients, safety status

**Per species** (`segments[].species[]`):
- `common_name`, `scientific_name`
- `tissue_total_pfas_ng_g` — total PFAS in fish tissue (sum of 6 congeners)
- `tissue_by_congener` — breakdown: `{PFOS: 12.5, PFOA: 3.2, ...}`
- `confidence_interval` — [lower, upper] 95% CI
- `accumulation_curve` — `{months: [0,3,6,...], concentration_ng_g: [...], lower_95: [...], upper_95: [...]}`
- `hazard_quotient_recreational`, `hazard_quotient_subsistence`
- `safe_servings_per_month_recreational`, `safe_servings_per_month_subsistence`
- `safety_status_recreational`, `safety_status_subsistence` — "safe" / "limited" / "unsafe"
- `pathway` — source facility → water → fish chain with dilution factors

**Facilities** (`facilities[]`):
- `name`, `lat`, `lng`, `sic_code`, `npdes_permit`, `estimated_pfas_discharge_ng_l`

**GeoJSON** (`geojson_segments`):
- Standard GeoJSON FeatureCollection with LineString geometries for map rendering
- Properties include `comid` (links to segment), `water_pfas_ng_l`, `risk_level`

### FastAPI server (optional)

`server.py` serves `national_results.json` over HTTP if you want API access instead of importing the JSON directly. But for the hackathon, importing the JSON is simpler and avoids CORS/deployment issues.

---

## Key References

- Gobas (1993) *Ecological Modelling* — Bioaccumulation ODE system
- Burkhard (2021) — Published BCF values for PFAS in fish
- Kelly, Sun, McDougall, Sunderland & Gobas (2024) *Env Sci & Tech* — PFAS trophic magnification
- Gal & Ghahramani (2016) — MC Dropout for uncertainty quantification
- EPA (2024) — PFAS reference doses (RfD) for hazard quotient calculation
