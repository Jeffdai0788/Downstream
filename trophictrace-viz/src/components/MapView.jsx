import { useRef, useEffect, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import riverGeo from '../data/riverGeometry.json'

mapboxgl.accessToken = 'pk.eyJ1IjoiamQxMjM0NTYiLCJhIjoiY21uYXR1dzdwMG43dTJwcHI0d2ltdXRzbCJ9.3tN6tOw4eqy-YGeGdU1Uhg'

const SAFE_COLOR = '#2EB872'
const LIMITED_COLOR = '#E0A030'
const MODERATE_COLOR = '#E8845A'
const UNSAFE_COLOR = '#DC4444'

// Green gradient for clean water (varies by zoom for depth)
const WATER_GREEN_DARK = '#0B2E1F'
const WATER_GREEN_MID = '#0F3D2A'
const WATER_GREEN_LIGHT = '#134A33'

export default function MapView({ data, onSegmentHover, onCursorMove, onMapReady, speciesFilter }) {
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
      ;['river-contamination', 'river-glow', 'river-hit-area'].forEach((id) => {
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

    ;['river-contamination', 'river-glow', 'river-hit-area'].forEach((id) => {
      if (map.current.getLayer(id)) map.current.setFilter(id, filter)
    })
  }, [speciesFilter, loaded, data])

  function getHotspotForSegment(seg) {
    const hotspots = [
      { id: 'cape_fear', lat: 35.05, lng: -78.88 },
      { id: 'lake_michigan', lat: 42.36, lng: -87.83 },
      { id: 'ohio_river', lat: 39.26, lng: -81.55 },
      { id: 'delaware_river', lat: 40.15, lng: -74.82 },
      { id: 'huron_river', lat: 42.28, lng: -83.74 },
      { id: 'merrimack_river', lat: 42.84, lng: -71.30 },
      { id: 'tennessee_river', lat: 34.58, lng: -86.96 },
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
      setLoaded(true)
      if (onMapReady) onMapReady(map.current)
    })
  }

  // --- Restyle all water to gradient green ---

  function restyleWaterLayer() {
    const m = map.current

    // Base water layer — green with zoom-dependent variation
    m.setPaintProperty('water', 'fill-color', [
      'interpolate', ['linear'], ['zoom'],
      3, WATER_GREEN_DARK,
      7, WATER_GREEN_MID,
      12, WATER_GREEN_LIGHT,
    ])
    m.setPaintProperty('water', 'fill-opacity', [
      'interpolate', ['linear'], ['zoom'],
      3, 0.85,
      8, 0.9,
      14, 0.95,
    ])

    // Also restyle any waterway line layers in the base style
    const style = m.getStyle()
    if (style && style.layers) {
      style.layers.forEach((layer) => {
        if (layer.id.includes('waterway') && layer.type === 'line') {
          m.setPaintProperty(layer.id, 'line-color', WATER_GREEN_MID)
          m.setPaintProperty(layer.id, 'line-opacity', 0.7)
        }
      })
    }
  }

  // --- River contamination layers ---

  function addRiverLayers() {
    const m = map.current

    m.addSource('contaminated-rivers', {
      type: 'geojson',
      data: riverGeo,
      tolerance: 0.375,
    })

    // Glow underlay — wider, blurred, lower opacity
    m.addLayer({
      id: 'river-glow',
      type: 'line',
      source: 'contaminated-rivers',
      paint: {
        'line-color': [
          'interpolate', ['linear'], ['get', 'pfas_ng_l'],
          30, SAFE_COLOR,
          200, LIMITED_COLOR,
          500, MODERATE_COLOR,
          900, UNSAFE_COLOR,
        ],
        'line-width': [
          'interpolate', ['exponential', 1.5], ['zoom'],
          4, ['interpolate', ['linear'], ['get', 'pfas_ng_l'], 30, 2, 500, 4, 1200, 6],
          8, ['interpolate', ['linear'], ['get', 'pfas_ng_l'], 30, 5, 500, 10, 1200, 16],
          12, ['interpolate', ['linear'], ['get', 'pfas_ng_l'], 30, 8, 500, 16, 1200, 28],
        ],
        'line-blur': [
          'interpolate', ['linear'], ['zoom'],
          4, 3,
          8, 6,
          12, 10,
        ],
        'line-opacity': [
          'interpolate', ['linear'], ['get', 'pfas_ng_l'],
          30, 0.08,
          200, 0.15,
          500, 0.22,
          900, 0.30,
        ],
      },
      layout: { 'line-cap': 'round', 'line-join': 'round' },
    })

    // Main contamination line
    m.addLayer({
      id: 'river-contamination',
      type: 'line',
      source: 'contaminated-rivers',
      paint: {
        'line-color': [
          'interpolate', ['linear'], ['get', 'pfas_ng_l'],
          30, SAFE_COLOR,
          200, LIMITED_COLOR,
          500, MODERATE_COLOR,
          900, UNSAFE_COLOR,
        ],
        'line-width': [
          'interpolate', ['exponential', 1.5], ['zoom'],
          4, ['interpolate', ['linear'], ['get', 'pfas_ng_l'], 30, 0.8, 500, 1.5, 1200, 2.5],
          8, ['interpolate', ['linear'], ['get', 'pfas_ng_l'], 30, 2, 500, 4, 1200, 7],
          12, ['interpolate', ['linear'], ['get', 'pfas_ng_l'], 30, 3, 500, 7, 1200, 12],
          14, ['interpolate', ['linear'], ['get', 'pfas_ng_l'], 30, 4, 500, 10, 1200, 18],
        ],
        'line-opacity': [
          'interpolate', ['linear'], ['get', 'pfas_ng_l'],
          30, 0.5,
          200, 0.7,
          500, 0.85,
          900, 0.95,
        ],
      },
      layout: { 'line-cap': 'round', 'line-join': 'round' },
    })

    // Invisible wide hit area for hover detection
    m.addLayer({
      id: 'river-hit-area',
      type: 'line',
      source: 'contaminated-rivers',
      paint: {
        'line-color': 'transparent',
        'line-width': [
          'interpolate', ['linear'], ['zoom'],
          4, 12,
          8, 20,
          12, 30,
        ],
        'line-opacity': 0,
      },
      layout: { 'line-cap': 'round', 'line-join': 'round' },
    })

    // Hover interaction
    m.on('mousemove', 'river-hit-area', (e) => {
      m.getCanvas().style.cursor = 'none'
      const props = e.features[0].properties
      const hotspotId = props.hotspot_id

      // Find the nearest data segment from nationalResults
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
        {/* Gradient bar */}
        <div style={{
          height: '8px', borderRadius: '4px', marginBottom: '0.375rem',
          background: `linear-gradient(to right, ${SAFE_COLOR}, ${LIMITED_COLOR}, ${MODERATE_COLOR}, ${UNSAFE_COLOR})`,
          opacity: 0.85,
        }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-tertiary)', fontSize: '0.625rem', fontFamily: 'var(--font-mono)' }}>
          <span>0</span>
          <span>200</span>
          <span>500</span>
          <span>900+</span>
        </div>
        {/* Water base color indicator */}
        <div style={{ marginTop: '0.625rem', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
          <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: WATER_GREEN_MID, border: '1px solid var(--border)' }} />
          <span style={{ color: 'var(--text-tertiary)', fontSize: '0.625rem' }}>Water bodies (no data = safe)</span>
        </div>
      </div>
    </div>
  )
}
