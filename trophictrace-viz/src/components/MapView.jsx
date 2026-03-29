import { useRef, useEffect, useState } from 'react'
import mapboxgl from 'mapbox-gl'

mapboxgl.accessToken = 'pk.eyJ1IjoiamQxMjM0NTYiLCJhIjoiY21uYXR1dzdwMG43dTJwcHI0d2ltdXRzbCJ9.3tN6tOw4eqy-YGeGdU1Uhg'

const SAFE_COLOR = '#2EB872'
const LIMITED_COLOR = '#E0A030'
const UNSAFE_COLOR = '#DC4444'

export default function MapView({ data, onSegmentHover, onCursorMove, onMapReady, speciesFilter }) {
  const mapContainer = useRef(null)
  const map = useRef(null)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    if (map.current) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !map.current) {
          observer.disconnect()
          initMap()
        }
      },
      { threshold: 0.1 }
    )
    observer.observe(mapContainer.current)

    return () => {
      observer.disconnect()
      if (map.current) map.current.remove()
    }
  }, [])

  // Apply species filter when it changes
  useEffect(() => {
    if (!map.current || !loaded) return

    const layerIds = ['river-glow', 'river-line', 'heatmap-glow', 'heatmap-core']

    if (!speciesFilter) {
      // Clear all filters
      layerIds.forEach((id) => {
        if (map.current.getLayer(id)) map.current.setFilter(id, null)
      })
      return
    }

    // Find comids for segments that contain this species
    const matchingComids = data.segments
      .filter((seg) => seg.species.some((sp) => sp.common_name === speciesFilter))
      .map((seg) => seg.comid)

    const filter = ['in', ['get', 'comid'], ['literal', matchingComids]]
    layerIds.forEach((id) => {
      if (map.current.getLayer(id)) map.current.setFilter(id, filter)
    })
  }, [speciesFilter, loaded, data])

  function initMap() {
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [-78.88, 35.05],
      zoom: 9.5,
      minZoom: 4,
      maxZoom: 14,
      attributionControl: false,
    })

    map.current.addControl(
      new mapboxgl.NavigationControl({ showCompass: false }),
      'bottom-right'
    )

    map.current.on('load', () => {
      map.current.resize()
      addHeatmapLayers()
      addRiverLayers()
      addFacilityMarkers()
      setLoaded(true)
      if (onMapReady) onMapReady(map.current)
    })
  }

  function addHeatmapLayers() {
    const m = map.current

    // Generate point features from segment centroids
    const heatmapPoints = {
      type: 'FeatureCollection',
      features: data.segments.map((seg) => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [seg.lng, seg.lat] },
        properties: {
          comid: seg.comid,
          water_pfas_ng_l: seg.predicted_water_pfas_ng_l,
          max_tissue_ng_g: Math.max(...seg.species.map((sp) => sp.tissue_total_pfas_ng_g)),
        },
      })),
    }

    m.addSource('heatmap-points', { type: 'geojson', data: heatmapPoints })

    // Outer glow circle
    m.addLayer({
      id: 'heatmap-glow',
      type: 'circle',
      source: 'heatmap-points',
      paint: {
        'circle-radius': [
          'interpolate', ['linear'], ['zoom'],
          8, ['interpolate', ['linear'], ['get', 'water_pfas_ng_l'], 0, 40, 120, 100],
          12, ['interpolate', ['linear'], ['get', 'water_pfas_ng_l'], 0, 80, 120, 180],
        ],
        'circle-color': [
          'interpolate', ['linear'], ['get', 'water_pfas_ng_l'],
          0, LIMITED_COLOR,
          60, '#E8845A',
          120, UNSAFE_COLOR,
        ],
        'circle-blur': 1,
        'circle-opacity': 0.12,
      },
    })

    // Inner core circle
    m.addLayer({
      id: 'heatmap-core',
      type: 'circle',
      source: 'heatmap-points',
      paint: {
        'circle-radius': [
          'interpolate', ['linear'], ['zoom'],
          8, ['interpolate', ['linear'], ['get', 'water_pfas_ng_l'], 0, 25, 120, 60],
          12, ['interpolate', ['linear'], ['get', 'water_pfas_ng_l'], 0, 50, 120, 100],
        ],
        'circle-color': [
          'interpolate', ['linear'], ['get', 'water_pfas_ng_l'],
          0, LIMITED_COLOR,
          60, '#E8845A',
          120, UNSAFE_COLOR,
        ],
        'circle-blur': 0.6,
        'circle-opacity': 0.08,
      },
    })
  }

  function addRiverLayers() {
    const m = map.current

    m.addSource('river', { type: 'geojson', data: data.geojson_segments })

    // Glow layer
    m.addLayer({
      id: 'river-glow',
      type: 'line',
      source: 'river',
      paint: {
        'line-color': [
          'interpolate', ['linear'], ['get', 'water_pfas_ng_l'],
          0, SAFE_COLOR, 60, LIMITED_COLOR, 120, UNSAFE_COLOR,
        ],
        'line-width': [
          'interpolate', ['linear'], ['get', 'water_pfas_ng_l'],
          0, 6, 120, 18,
        ],
        'line-opacity': 0.15,
        'line-blur': 8,
      },
    })

    // Main river line
    m.addLayer({
      id: 'river-line',
      type: 'line',
      source: 'river',
      paint: {
        'line-color': [
          'interpolate', ['linear'], ['get', 'water_pfas_ng_l'],
          0, SAFE_COLOR, 60, LIMITED_COLOR, 120, UNSAFE_COLOR,
        ],
        'line-width': [
          'interpolate', ['linear'], ['get', 'water_pfas_ng_l'],
          0, 2, 120, 5,
        ],
        'line-opacity': 0.85,
      },
      layout: { 'line-cap': 'round', 'line-join': 'round' },
    })

    // Hover: hide default cursor (custom bubble in FloatingIsland)
    m.on('mousemove', 'river-line', (e) => {
      m.getCanvas().style.cursor = 'none'
      const props = e.features[0].properties
      const comid = props.comid
      const segment = data.segments.find((s) => s.comid === comid)
      if (segment) {
        onSegmentHover(segment, e)
        onCursorMove({ x: e.point.x, y: e.point.y })
      }
    })

    m.on('mouseleave', 'river-line', () => {
      m.getCanvas().style.cursor = ''
      onSegmentHover(null)
    })
  }

  function addFacilityMarkers() {
    data.facilities.forEach((facility) => {
      const el = document.createElement('div')
      el.innerHTML = `
        <div style="
          width: 10px; height: 10px; background: var(--accent);
          border-radius: 50%; border: 1.5px solid var(--text-primary);
          box-shadow: 0 0 12px rgba(212, 145, 110, 0.4);
        "></div>
      `

      const sectorLabel = facility.pfas_sector ? 'PFAS sector' : 'Non-PFAS sector'

      const popup = new mapboxgl.Popup({ offset: 12, closeButton: false, className: 'facility-popup' })
        .setHTML(`
          <div style="font-family: var(--font-body); font-size: 12px; color: var(--text-primary); padding: 4px 0;">
            <div style="font-weight: 500; margin-bottom: 4px;">${facility.name}</div>
            <div style="color: var(--text-secondary); font-family: var(--font-mono); font-size: 11px; margin-bottom: 2px;">
              ${facility.estimated_pfas_discharge_ng_l} ng/L discharge
            </div>
            <div style="color: var(--text-tertiary); font-size: 10px;">
              NPDES: ${facility.npdes_permit} | ${sectorLabel}
            </div>
          </div>
        `)

      new mapboxgl.Marker(el)
        .setLngLat([facility.lng, facility.lat])
        .setPopup(popup)
        .addTo(map.current)
    })
  }

  return (
    <div style={{ position: 'absolute', inset: 0, background: 'var(--bg-primary)' }}>
      {/* Title bar */}
      <div
        style={{
          position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10,
          padding: '1rem 1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          background: 'linear-gradient(to bottom, rgba(25,25,25,0.8) 0%, transparent 100%)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '1rem' }}>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: '1.125rem', fontWeight: 500, color: 'var(--text-primary)' }}>
            TrophicTrace
          </span>
          <span style={{ fontFamily: 'var(--font-body)', fontSize: '0.8125rem', color: 'var(--text-tertiary)' }}>
            Cape Fear River, NC
          </span>
        </div>
      </div>

      <div ref={mapContainer} style={{ width: '100%', height: '100%' }} />

      {/* Legend */}
      <div
        style={{
          position: 'absolute', bottom: '2rem', left: '1.5rem', zIndex: 10,
          background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: '12px',
          padding: '0.875rem 1.125rem', fontFamily: 'var(--font-body)', fontSize: '0.75rem',
        }}
      >
        <div style={{ color: 'var(--text-secondary)', fontWeight: 500, marginBottom: '0.5rem', fontSize: '0.6875rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Fish Tissue PFAS (ng/g)
        </div>
        <div style={{ display: 'flex', gap: '0.875rem', alignItems: 'center' }}>
          {[
            { color: SAFE_COLOR, label: '< 5 Safe' },
            { color: LIMITED_COLOR, label: '5-20 Limited' },
            { color: UNSAFE_COLOR, label: '> 20 Unsafe' },
          ].map(({ color, label }) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
              <div style={{ width: '10px', height: '10px', borderRadius: '2px', background: color }} />
              <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
