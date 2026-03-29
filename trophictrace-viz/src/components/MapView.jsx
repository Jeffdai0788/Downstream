import { useRef, useEffect, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import riverGeo from '../data/riverGeometry.json'
import { ENV_CENTERS } from '../data/envCenters'

mapboxgl.accessToken = 'pk.eyJ1IjoiamQxMjM0NTYiLCJhIjoiY21uYXR1dzdwMG43dTJwcHI0d2ltdXRzbCJ9.3tN6tOw4eqy-YGeGdU1Uhg'

const SAFE_COLOR = '#2EB872'
const LIMITED_COLOR = '#E0A030'
const MODERATE_COLOR = '#E8845A'
const UNSAFE_COLOR = '#DC4444'

// Slight blue tint for water bodies
const WATER_TINT = '#4A90D9'

export default function MapView({ data, onSegmentHover, onSegmentClick, onCursorMove, onMapReady, speciesFilter }) {
  const mapContainer = useRef(null)
  const map = useRef(null)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    if (map.current) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !map.current) { observer.disconnect(); initMap() }
      },
      { threshold: 0.1 }
    )
    observer.observe(mapContainer.current)
    return () => { observer.disconnect(); if (map.current) map.current.remove() }
  }, [])

  // Species filter — filter river segments to only show where that species lives
  useEffect(() => {
    if (!map.current || !loaded) return

    if (!speciesFilter) {
      ;['river-plume', 'river-contamination', 'river-hit-area'].forEach((id) => {
        if (map.current.getLayer(id)) map.current.setFilter(id, null)
      })
      return
    }

    // Find which hotspots contain this species
    const matchingHotspots = new Set()
    data.segments.forEach((seg) => {
      if (seg.species.some((sp) => sp.common_name === speciesFilter)) {
        // Map segment location to nearest hotspot
        const hotspot = getHotspotForSegment(seg)
        if (hotspot) matchingHotspots.add(hotspot)
      }
    })

    const filter = matchingHotspots.size > 0
      ? ['in', ['get', 'hotspot_id'], ['literal', [...matchingHotspots]]]
      : ['==', ['get', 'hotspot_id'], '__none__']

    ;['river-plume', 'river-contamination', 'river-hit-area'].forEach((id) => {
      if (map.current.getLayer(id)) map.current.setFilter(id, filter)
    })
  }, [speciesFilter, loaded, data])

  function getHotspotForSegment(seg) {
    const hotspots = [
      { id: 'cape_fear',          lat: 35.05, lng: -78.88 },
      { id: 'lake_michigan',      lat: 42.36, lng: -87.82 },
      { id: 'ohio_river',         lat: 39.26, lng: -81.55 },
      { id: 'delaware_river',     lat: 40.15, lng: -74.82 },
      { id: 'huron_river',        lat: 42.28, lng: -83.74 },
      { id: 'merrimack_river',    lat: 42.84, lng: -71.30 },
      { id: 'tennessee_river',    lat: 34.58, lng: -86.96 },
      { id: 'cape_fear_atlantic', lat: 33.65, lng: -77.70 },
      { id: 'delaware_atlantic',  lat: 38.50, lng: -74.70 },
      { id: 'gulf_mexico',        lat: 28.80, lng: -89.20 },
      { id: 'lake_erie',          lat: 41.78, lng: -83.20 },
      { id: 'lake_ontario',       lat: 43.30, lng: -79.30 },
      { id: 'merrimack_atlantic', lat: 42.82, lng: -70.55 },
    ]
    let minDist = Infinity
    let nearest = null
    for (const h of hotspots) {
      const d = Math.hypot(seg.latitude - h.lat, seg.longitude - h.lng)
      if (d < minDist) { minDist = d; nearest = h.id }
    }
    return minDist < 1.0 ? nearest : null
  }

  function initMap() {
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [-83, 38],
      zoom: 4.2,
      minZoom: 3,
      maxZoom: 14,
      attributionControl: false,
    })

    map.current.addControl(new mapboxgl.NavigationControl({ showCompass: false }), 'bottom-right')

    map.current.on('load', () => {
      map.current.resize()
      restyleWaterLayer()
      addRiverLayers()
      addFacilityMarkers()
      addEnvCenterMarkers()
      setLoaded(true)
      if (onMapReady) onMapReady(map.current)
    })
  }

  // --- Translucent blue tint on water bodies ---

  function restyleWaterLayer() {
    const m = map.current

    m.setPaintProperty('water', 'fill-color', WATER_TINT)
    m.setPaintProperty('water', 'fill-opacity', 0.18)

    const style = m.getStyle()
    if (style && style.layers) {
      style.layers.forEach((layer) => {
        if (layer.id.includes('waterway') && layer.type === 'line') {
          m.setPaintProperty(layer.id, 'line-color', WATER_TINT)
          m.setPaintProperty(layer.id, 'line-opacity', 0.25)
        }
      })
    }
  }

  // --- Contamination plume + flow layers ---

  // Extract midpoint of each LineString as a weighted Point for the heatmap
  function buildPlumePoints() {
    return {
      type: 'FeatureCollection',
      features: riverGeo.features.map((feat) => {
        const coords = feat.geometry.coordinates
        const mid = coords[Math.floor(coords.length / 2)]
        return { type: 'Feature', properties: feat.properties, geometry: { type: 'Point', coordinates: mid } }
      }),
    }
  }

  function addRiverLayers() {
    const m = map.current

    // Source for line hover hit-area + flow centerline
    m.addSource('contaminated-rivers', {
      type: 'geojson',
      data: riverGeo,
      tolerance: 0.375,
    })

    // Source for heatmap blobs (points at LineString midpoints)
    m.addSource('plume-points', {
      type: 'geojson',
      data: buildPlumePoints(),
    })

    // Insert heatmap BELOW the first road layer so roads/buildings on land
    // naturally render on top of it, limiting visible bleed over developed land
    const styleLayers = m.getStyle().layers
    const firstRoadLayer = styleLayers.find((l) => l.id.startsWith('road-') || l.id === 'road')
    const beforeRoadId = firstRoadLayer?.id

    // ── Radiating plume blobs ───────────────────────────────────────────────
    m.addLayer({
      id: 'river-plume',
      type: 'heatmap',
      source: 'plume-points',
      paint: {
        // Weight each point by its PFAS level so high-contamination sources
        // accumulate higher density → warmer color
        'heatmap-weight': [
          'interpolate', ['linear'], ['get', 'pfas_ng_l'],
          0,    0,
          150,  0.22,
          500,  0.55,
          900,  0.80,
          1200, 1.0,
        ],
        // Radius: large enough so adjacent river-segment points merge into one
        // continuous organic blob; shrinks at high zoom where detail takes over
        'heatmap-radius': [
          'interpolate', ['linear'], ['zoom'],
          3,  10,
          5,  20,
          7,  32,
          9,  50,
          11, 68,
          13, 90,
        ],
        // Intensity: controls how quickly density accumulates in overlapping areas
        'heatmap-intensity': [
          'interpolate', ['linear'], ['zoom'],
          3,  0.4,
          6,  0.8,
          9,  1.5,
          13, 3.0,
        ],
        // Color ramp: transparent at edges → safe green → amber → orange → red at core
        'heatmap-color': [
          'interpolate', ['linear'], ['heatmap-density'],
          0,    'rgba(0,0,0,0)',
          0.05, 'rgba(0,0,0,0)',
          0.15, `rgba(46,184,114,0.45)`,
          0.35, `rgba(46,184,114,0.72)`,
          0.50, `rgba(224,160,48,0.80)`,
          0.68, `rgba(232,132,90,0.88)`,
          0.85, `rgba(220,68,68,0.92)`,
          1.0,  `rgba(200,40,40,0.97)`,
        ],
        // Fade slightly at very high zoom — the thin centerline takes over
        'heatmap-opacity': [
          'interpolate', ['linear'], ['zoom'],
          3,  0.80,
          8,  0.88,
          11, 0.75,
          14, 0.45,
        ],
      },
    }, beforeRoadId)

    // ── Thin flow centerline (subtle reference path) ────────────────────────
    const ZONE_COLOR = [
      'step', ['get', 'pfas_ng_l'],
      SAFE_COLOR,
      150, LIMITED_COLOR,
      500, MODERATE_COLOR,
      900, UNSAFE_COLOR,
    ]

    m.addLayer({
      id: 'river-contamination',
      type: 'line',
      source: 'contaminated-rivers',
      paint: {
        'line-color': ZONE_COLOR,
        'line-width': [
          'interpolate', ['exponential', 1.5], ['zoom'],
          4,  0.3,
          8,  0.7,
          12, 1.5,
          14, 2.5,
        ],
        'line-opacity': [
          'interpolate', ['linear'], ['zoom'],
          4,  0.0,
          7,  0.25,
          11, 0.50,
          14, 0.70,
        ],
      },
      layout: { 'line-cap': 'round', 'line-join': 'round' },
    })

    // ── Invisible wide hit area for hover detection ─────────────────────────
    m.addLayer({
      id: 'river-hit-area',
      type: 'line',
      source: 'contaminated-rivers',
      paint: {
        'line-color': 'transparent',
        'line-width': ['interpolate', ['linear'], ['zoom'], 4, 12, 8, 20, 12, 30],
        'line-opacity': 0,
      },
      layout: { 'line-cap': 'round', 'line-join': 'round' },
    })

    m.on('mousemove', 'river-hit-area', (e) => {
      m.getCanvas().style.cursor = 'none'
      const point = e.lngLat
      let nearest = null
      let minDist = Infinity
      for (const seg of data.segments) {
        const d = Math.hypot(seg.latitude - point.lat, seg.longitude - point.lng)
        if (d < minDist) { minDist = d; nearest = seg }
      }
      if (nearest && minDist < 0.5) {
        onSegmentHover(nearest, e)
        onCursorMove({ x: e.point.x, y: e.point.y })
      }
    })

    m.on('mouseleave', 'river-hit-area', () => {
      m.getCanvas().style.cursor = ''
      onSegmentHover(null)
    })

    m.on('click', 'river-hit-area', (e) => {
      const point = e.lngLat
      let nearest = null
      let minDist = Infinity
      for (const seg of data.segments) {
        const d = Math.hypot(seg.latitude - point.lat, seg.longitude - point.lng)
        if (d < minDist) { minDist = d; nearest = seg }
      }
      if (nearest && minDist < 0.5 && onSegmentClick) {
        onSegmentClick(nearest)
      }
    })
  }

  function addEnvCenterMarkers() {
    const TYPE_COLOR = { epa_regional: '#06B6D4', state_pfas: '#3B82F6', national_hotline: '#8B5CF6' }
    ENV_CENTERS.forEach((center) => {
      const color = TYPE_COLOR[center.type] ?? '#888'
      const el = document.createElement('div')
      el.title = center.name
      el.innerHTML = `<div style="
        width: 10px; height: 10px;
        background: ${color};
        border: 1.5px solid rgba(255,255,255,0.5);
        transform: rotate(45deg);
        box-shadow: 0 0 8px ${color}88;
        cursor: default;
      "></div>`
      const popup = new mapboxgl.Popup({ offset: 12, closeButton: false })
        .setHTML(`<div style="font-family: var(--font-body); font-size: 12px; color: var(--text-primary); padding: 4px 0;">
          <div style="font-weight: 500; margin-bottom: 2px;">${center.name}</div>
          <div style="color: ${color}; font-size: 10px; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.05em;">${center.type.replace('_', ' ')}</div>
          <div style="color: var(--text-tertiary); font-size: 10px;">${center.phone}</div>
        </div>`)
      new mapboxgl.Marker(el).setLngLat([center.lng, center.lat]).setPopup(popup).addTo(map.current)
    })
  }

  function addFacilityMarkers() {
    data.facilities.forEach((facility) => {
      const el = document.createElement('div')
      el.innerHTML = `<div style="
        width: 10px; height: 10px; background: var(--accent);
        border-radius: 50%; border: 1.5px solid var(--text-primary);
        box-shadow: 0 0 12px rgba(212, 145, 110, 0.4);
      "></div>`

      const popup = new mapboxgl.Popup({ offset: 12, closeButton: false })
        .setHTML(`<div style="font-family: var(--font-body); font-size: 12px; color: var(--text-primary); padding: 4px 0;">
          <div style="font-weight: 500; margin-bottom: 2px;">${facility.name}</div>
          <div style="color: var(--text-tertiary); font-size: 10px;">${facility.pfas_sector ? 'PFAS sector' : 'Non-PFAS sector'}</div>
        </div>`)

      new mapboxgl.Marker(el).setLngLat([facility.lng, facility.lat]).setPopup(popup).addTo(map.current)
    })
  }

  return (
    <div style={{ position: 'absolute', inset: 0, background: 'var(--bg-primary)' }}>
      {/* Title bar */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10,
        padding: '1rem 1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: 'linear-gradient(to bottom, rgba(25,25,25,0.8) 0%, transparent 100%)',
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '1rem' }}>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: '1.125rem', fontWeight: 500, color: 'var(--text-primary)' }}>TrophicTrace</span>
          <span style={{ fontFamily: 'var(--font-body)', fontSize: '0.8125rem', color: 'var(--text-tertiary)' }}>National PFAS Risk Map</span>
        </div>
      </div>

      <div ref={mapContainer} style={{ width: '100%', height: '100%' }} />

      {/* Legend — gradient bar */}
      <div style={{
        position: 'absolute', bottom: '2rem', left: '1.5rem', zIndex: 10,
        background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: '12px',
        padding: '0.875rem 1.125rem', fontFamily: 'var(--font-body)', fontSize: '0.75rem',
      }}>
        <div style={{ color: 'var(--text-secondary)', fontWeight: 500, marginBottom: '0.625rem', fontSize: '0.6875rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Water PFAS Contamination (ng/L)
        </div>
        {/* Plume gradient bar */}
        <div style={{
          height: '8px', borderRadius: '4px', marginBottom: '0.4rem',
          background: `linear-gradient(to right, ${SAFE_COLOR}, ${LIMITED_COLOR}, ${MODERATE_COLOR}, ${UNSAFE_COLOR})`,
          opacity: 0.88,
        }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-tertiary)', fontSize: '0.6rem', fontFamily: 'var(--font-mono)', marginBottom: '0.25rem' }}>
          <span>0</span><span>150</span><span>500</span><span>900+</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-tertiary)', fontSize: '0.6rem' }}>
          <span>Safe</span><span>Limited</span><span>Moderate</span><span>Unsafe</span>
        </div>
        {/* Water tint indicator */}
        <div style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
          <div style={{ width: '28px', height: '8px', borderRadius: '2px', flexShrink: 0, background: WATER_TINT, opacity: 0.4, border: '1px solid var(--border)' }} />
          <span style={{ color: 'var(--text-tertiary)', fontSize: '0.625rem' }}>Unsampled water</span>
        </div>
      </div>
    </div>
  )
}
