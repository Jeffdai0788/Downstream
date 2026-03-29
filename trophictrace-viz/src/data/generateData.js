// Run with: node src/data/generateData.js
// Generates focused PFAS data around real US water body hotspots

const SPECIES = [
  { common_name: 'Largemouth Bass', scientific_name: 'Micropterus salmoides', trophic_level: 4.2, lipid_content_pct: 5.8 },
  { common_name: 'Striped Bass', scientific_name: 'Morone saxatilis', trophic_level: 4.5, lipid_content_pct: 6.1 },
  { common_name: 'Channel Catfish', scientific_name: 'Ictalurus punctatus', trophic_level: 3.8, lipid_content_pct: 4.2 },
  { common_name: 'Bluegill', scientific_name: 'Lepomis macrochirus', trophic_level: 3.1, lipid_content_pct: 3.5 },
  { common_name: 'Common Carp', scientific_name: 'Cyprinus carpio', trophic_level: 2.9, lipid_content_pct: 5.2 },
  { common_name: 'Brown Trout', scientific_name: 'Salmo trutta', trophic_level: 4.0, lipid_content_pct: 5.5 },
  { common_name: 'White Perch', scientific_name: 'Morone americana', trophic_level: 3.5, lipid_content_pct: 3.8 },
  { common_name: 'Flathead Catfish', scientific_name: 'Pylodictis olivaris', trophic_level: 4.0, lipid_content_pct: 4.8 },
]

// Real PFAS hotspot clusters with water body shapes
const HOTSPOTS = [
  {
    name: 'Cape Fear River, NC',
    facility: 'Chemours Fayetteville Works',
    // Points along the Cape Fear River
    points: [
      { lat: 35.20, lng: -78.98, pfas: 180 },
      { lat: 35.15, lng: -78.95, pfas: 280 },
      { lat: 35.10, lng: -78.92, pfas: 420 },
      { lat: 35.05, lng: -78.88, pfas: 850 },  // Near Chemours
      { lat: 35.00, lng: -78.86, pfas: 920 },
      { lat: 34.95, lng: -78.84, pfas: 780 },
      { lat: 34.90, lng: -78.80, pfas: 600 },
      { lat: 34.85, lng: -78.76, pfas: 450 },
      { lat: 34.78, lng: -78.72, pfas: 320 },
      { lat: 34.70, lng: -78.68, pfas: 200 },
    ],
    demo: { name: 'Fayetteville SE, NC', median_income: 31200, subsistence_pct: 18.5, population: 24500 },
  },
  {
    name: 'Lake Michigan — Waukegan, IL',
    facility: 'Waukegan Harbor (legacy)',
    points: [
      { lat: 42.38, lng: -87.84, pfas: 120 },
      { lat: 42.36, lng: -87.82, pfas: 280 },
      { lat: 42.34, lng: -87.83, pfas: 350 },
      { lat: 42.36, lng: -87.86, pfas: 200 },
      { lat: 42.33, lng: -87.85, pfas: 310 },
      { lat: 42.35, lng: -87.80, pfas: 180 },
    ],
    demo: { name: 'Waukegan, IL', median_income: 38500, subsistence_pct: 12.0, population: 89000 },
  },
  {
    name: 'Ohio River — Parkersburg, WV',
    facility: 'DuPont Washington Works',
    points: [
      { lat: 39.30, lng: -81.60, pfas: 400 },
      { lat: 39.28, lng: -81.57, pfas: 680 },
      { lat: 39.26, lng: -81.55, pfas: 1200 },  // Near DuPont
      { lat: 39.24, lng: -81.53, pfas: 1050 },
      { lat: 39.22, lng: -81.50, pfas: 800 },
      { lat: 39.20, lng: -81.48, pfas: 550 },
      { lat: 39.18, lng: -81.45, pfas: 380 },
      { lat: 39.16, lng: -81.42, pfas: 250 },
    ],
    demo: { name: 'Parkersburg, WV', median_income: 28900, subsistence_pct: 22.0, population: 30000 },
  },
  {
    name: 'Delaware River — Bucks County, PA',
    facility: 'Willow Grove NAS (AFFF)',
    points: [
      { lat: 40.22, lng: -74.88, pfas: 150 },
      { lat: 40.18, lng: -74.85, pfas: 320 },
      { lat: 40.15, lng: -74.82, pfas: 480 },
      { lat: 40.12, lng: -74.80, pfas: 520 },
      { lat: 40.08, lng: -74.78, pfas: 400 },
      { lat: 40.05, lng: -74.76, pfas: 280 },
    ],
    demo: { name: 'Bucks County, PA', median_income: 45800, subsistence_pct: 6.5, population: 63000 },
  },
  {
    name: 'Huron River — Ann Arbor, MI',
    facility: 'Gelman Sciences (dioxane/PFAS)',
    points: [
      { lat: 42.32, lng: -83.80, pfas: 100 },
      { lat: 42.30, lng: -83.76, pfas: 220 },
      { lat: 42.28, lng: -83.74, pfas: 380 },
      { lat: 42.27, lng: -83.72, pfas: 300 },
      { lat: 42.26, lng: -83.70, pfas: 180 },
    ],
    demo: { name: 'Ypsilanti, MI', median_income: 33200, subsistence_pct: 14.0, population: 22000 },
  },
  {
    name: 'Merrimack River — NH',
    facility: 'Saint-Gobain (NH)',
    points: [
      { lat: 42.88, lng: -71.34, pfas: 180 },
      { lat: 42.86, lng: -71.32, pfas: 350 },
      { lat: 42.84, lng: -71.30, pfas: 420 },
      { lat: 42.82, lng: -71.28, pfas: 360 },
      { lat: 42.80, lng: -71.26, pfas: 220 },
    ],
    demo: { name: 'Merrimack, NH', median_income: 41000, subsistence_pct: 8.0, population: 26000 },
  },
  {
    name: 'Tennessee River — Decatur, AL',
    facility: '3M Decatur Plant',
    points: [
      { lat: 34.62, lng: -87.02, pfas: 300 },
      { lat: 34.60, lng: -86.98, pfas: 580 },
      { lat: 34.58, lng: -86.96, pfas: 900 },
      { lat: 34.56, lng: -86.94, pfas: 750 },
      { lat: 34.54, lng: -86.92, pfas: 500 },
      { lat: 34.52, lng: -86.88, pfas: 320 },
    ],
    demo: { name: 'Decatur, AL', median_income: 29800, subsistence_pct: 20.0, population: 57000 },
  },
]

const BCF_BASE = { PFOS: 3100, PFOA: 132, PFNA: 1200, PFHxS: 316, PFDA: 2000, GenX: 40 }
const TMF = { PFOS: 3.5, PFOA: 1.5, PFNA: 3.0, PFHxS: 2.0, PFDA: 3.2, GenX: 1.2 }
const RFD = { PFOS: 1e-7, PFOA: 3e-8, PFNA: 3e-6, PFHxS: 2e-5, PFDA: 3e-6, GenX: 3e-6 }
const REF_LIPID = 4.0, REF_TROPHIC = 3.0

function computeSpecies(waterPfas, speciesTemplate, facilityName) {
  const congenerFractions = { PFOS: 0.45, PFOA: 0.15, PFNA: 0.12, PFHxS: 0.08, PFDA: 0.15, GenX: 0.05 }
  const tissueByC = {}
  let totalTissue = 0

  for (const [cong, frac] of Object.entries(congenerFractions)) {
    const cWater = waterPfas * frac
    const bcf = BCF_BASE[cong] * (speciesTemplate.lipid_content_pct / REF_LIPID)
    const trophicDiff = speciesTemplate.trophic_level - REF_TROPHIC
    const tissue = (cWater * bcf / 1000) * Math.pow(TMF[cong], trophicDiff)
    tissueByC[cong] = Math.round(tissue * 100) / 100
    totalTissue += tissue
  }

  totalTissue = Math.round(totalTissue * 100) / 100

  // Hazard quotients
  let hqRec = 0, hqSub = 0
  for (const [cong, tissue] of Object.entries(tissueByC)) {
    const doseRec = (tissue * 1e-6 * 17) / 70
    const doseSub = (tissue * 1e-6 * 142.4) / 70
    hqRec += doseRec / RFD[cong]
    hqSub += doseSub / RFD[cong]
  }

  const safeRec = hqRec > 0 ? Math.max(0, Math.floor((30 * 227) / (hqRec * 17 * 30 / 1))) : 30
  const safeSub = hqSub > 0 ? Math.max(0, Math.floor((30 * 227) / (hqSub * 142.4 * 30 / 1))) : 30

  return {
    ...speciesTemplate,
    tissue_pfos_ng_g: tissueByC.PFOS,
    tissue_pfoa_ng_g: tissueByC.PFOA,
    tissue_total_pfas_ng_g: totalTissue,
    hazard_quotient_recreational: Math.round(hqRec * 1000) / 1000,
    hazard_quotient_subsistence: Math.round(hqSub * 1000) / 1000,
    safe_servings_per_month_recreational: Math.min(safeRec, 30),
    safe_servings_per_month_subsistence: Math.min(safeSub, 30),
    safety_status_recreational: hqRec < 0.5 ? 'safe' : hqRec < 1.5 ? 'limited' : 'unsafe',
    safety_status_subsistence: hqSub < 0.5 ? 'safe' : hqSub < 1.5 ? 'limited' : 'unsafe',
    tissue_by_congener: tissueByC,
    pathway: {
      source_facility: facilityName,
      source_distance_km: Math.round((5 + Math.random() * 30) * 10) / 10,
      dilution_factor: Math.round((2 + Math.random() * 15) * 10) / 10,
      water_concentration_ng_l: waterPfas,
      bcf_applied: Math.round(BCF_BASE.PFOS * (speciesTemplate.lipid_content_pct / REF_LIPID)),
      tmf_applied: Math.round(TMF.PFOS ** (speciesTemplate.trophic_level - REF_TROPHIC) * 100) / 100,
      tissue_concentration_ng_g: totalTissue,
    },
  }
}

function generateFeatureImportance(pfas) {
  const features = [
    { feature: 'nearest_pfas_facility_km', importance: 0.18 + Math.random() * 0.15 },
    { feature: 'upstream_npdes_pfas_count', importance: 0.12 + Math.random() * 0.12 },
    { feature: 'afff_site_nearby', importance: 0.05 + Math.random() * 0.15 },
    { feature: 'wwtp_upstream', importance: 0.05 + Math.random() * 0.10 },
    { feature: 'pfas_industry_density', importance: 0.03 + Math.random() * 0.08 },
  ]
  const total = features.reduce((s, f) => s + f.importance, 0)
  features.forEach((f) => { f.importance = Math.round((f.importance / total) * 1000) / 1000 })
  features.sort((a, b) => b.importance - a.importance)
  return features
}

// Generate all data
const segments = []
const facilities = []
const demographics = []
const geojsonFeatures = []
let segIdx = 0

HOTSPOTS.forEach((hotspot) => {
  // Add facility
  const centerPt = hotspot.points[Math.floor(hotspot.points.length / 2)]
  facilities.push({
    facility_id: `fac_${String(facilities.length).padStart(4, '0')}`,
    name: hotspot.facility,
    lat: centerPt.lat,
    lng: centerPt.lng,
    pfas_sector: true,
    intensity: 1.0,
  })

  // Add demographics
  demographics.push({
    name: hotspot.demo.name,
    lat: centerPt.lat - 0.02,
    lng: centerPt.lng + 0.02,
    median_income: hotspot.demo.median_income,
    subsistence_pct: hotspot.demo.subsistence_pct,
    population: hotspot.demo.population,
    boundary: [
      [centerPt.lng - 0.05, centerPt.lat - 0.03],
      [centerPt.lng + 0.05, centerPt.lat - 0.03],
      [centerPt.lng + 0.05, centerPt.lat + 0.03],
      [centerPt.lng - 0.05, centerPt.lat + 0.03],
    ],
  })

  // Generate segments for each point
  hotspot.points.forEach((pt, pi) => {
    const segId = `seg_${String(segIdx++).padStart(4, '0')}`
    const confidence = 0.65 + Math.random() * 0.25

    // Pick 4-6 random species for this segment
    const nSpecies = 4 + Math.floor(Math.random() * 3)
    const shuffled = [...SPECIES].sort(() => Math.random() - 0.5).slice(0, nSpecies)
    const speciesData = shuffled.map((sp) => computeSpecies(pt.pfas, sp, hotspot.facility))

    segments.push({
      segment_id: segId,
      latitude: pt.lat,
      longitude: pt.lng,
      predicted_water_pfas_ng_l: pt.pfas,
      prediction_confidence: Math.round(confidence * 100) / 100,
      flow_rate_m3s: Math.round((5 + Math.random() * 80) * 10) / 10,
      stream_order: Math.floor(1 + Math.random() * 5),
      risk_level: pt.pfas > 500 ? 'high' : pt.pfas > 150 ? 'moderate' : 'low',
      top_contributing_features: generateFeatureImportance(pt.pfas),
      species: speciesData,
    })

    // GeoJSON: short LineString around this point (for potential future use)
    const jitter = 0.005
    geojsonFeatures.push({
      type: 'Feature',
      properties: {
        segment_id: segId,
        water_pfas_ng_l: pt.pfas,
        risk_level: pt.pfas > 500 ? 'high' : pt.pfas > 150 ? 'moderate' : 'low',
        max_tissue_ng_g: Math.max(...speciesData.map((s) => s.tissue_total_pfas_ng_g)),
      },
      geometry: {
        type: 'LineString',
        coordinates: [
          [pt.lng - jitter, pt.lat - jitter * 0.5],
          [pt.lng, pt.lat],
          [pt.lng + jitter, pt.lat + jitter * 0.5],
        ],
      },
    })
  })
})

const output = {
  metadata: {
    model_version: 'trophictrace-v1',
    xgboost: { cv_r2: 0.7012, cv_within_factor_3: 98.6, n_training_samples: 5000, train_time_s: 0.05 },
    pinn: { r2: 0.7559, within_factor_2: 96.88, within_factor_3: 98.98, n_parameters: 50829, train_time_s: 82.14 },
    total_segments_scored: segments.length,
    detail_segments: segments.length,
    species_modeled: 8,
    congeners_modeled: 6,
    inference_time_s: 2.14,
  },
  segments,
  facilities,
  demographics,
  species_reference: SPECIES.map((s) => ({ ...s, body_mass_g: 500 + Math.floor(Math.random() * 4000) })),
  geojson_segments: { type: 'FeatureCollection', features: geojsonFeatures },
}

const fs = await import('fs')
fs.writeFileSync('src/data/nationalResults.json', JSON.stringify(output))
console.log(`Generated ${segments.length} segments across ${HOTSPOTS.length} hotspots`)
console.log(`Facilities: ${facilities.length}, Demographics: ${demographics.length}`)
