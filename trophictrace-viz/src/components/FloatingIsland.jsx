import { useEffect, useState, useRef } from 'react'

const STATUS_COLORS = {
  safe: '#2EB872',
  limited: '#E0A030',
  unsafe: '#DC4444',
}

// Find nearest demographic zone to a segment
function findNearestDemographic(segment, demographics) {
  if (!demographics || demographics.length === 0) return null
  let nearest = null
  let minDist = Infinity
  demographics.forEach((d) => {
    const dist = Math.sqrt((segment.latitude - d.lat) ** 2 + (segment.longitude - d.lng) ** 2)
    if (dist < minDist && dist < 0.5) {
      minDist = dist
      nearest = d
    }
  })
  return nearest
}

export default function FloatingIsland({ segment, demographics, position, onSpeciesClick }) {
  const [islandPos, setIslandPos] = useState({ x: 0, y: 0 })
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    if (!segment) { setVisible(false); return }

    const islandWidth = 360
    const islandHeight = 400
    let x = position.x + 40
    let y = position.y + 80

    if (x + islandWidth > window.innerWidth - 20) x = position.x - islandWidth - 40
    if (y + islandHeight > window.innerHeight - 20) y = window.innerHeight - islandHeight - 20

    setIslandPos({ x, y })
    setVisible(true)
  }, [segment, position])

  if (!segment) return null

  const sorted = [...segment.species].sort(
    (a, b) => b.tissue_total_pfas_ng_g - a.tissue_total_pfas_ng_g
  )

  const nearestDemo = findNearestDemographic(segment, demographics)
  const anchorX = islandPos.x + 20
  const anchorY = islandPos.y + 10

  return (
    <>
      {/* Cursor bubble */}
      <div
        style={{
          position: 'fixed', left: position.x, top: position.y,
          width: '28px', height: '28px', borderRadius: '50%',
          background: 'rgba(160, 160, 160, 0.25)', border: '1px solid rgba(160, 160, 160, 0.15)',
          transform: 'translate(-50%, -50%)', pointerEvents: 'none', zIndex: 18,
          transition: 'opacity 150ms ease-out', opacity: visible ? 1 : 0,
        }}
      />

      {/* SVG connector */}
      <svg style={{ position: 'fixed', inset: 0, width: '100vw', height: '100vh', pointerEvents: 'none', zIndex: 19 }}>
        <line x1={position.x} y1={position.y} x2={anchorX} y2={anchorY}
          stroke="var(--text-tertiary)" strokeWidth={1} strokeDasharray="4 4" opacity={0.4} />
      </svg>

      {/* Island card */}
      <div
        style={{
          position: 'fixed', left: islandPos.x, top: islandPos.y, width: 360,
          background: 'rgba(25, 25, 25, 0.94)', backdropFilter: 'blur(12px)',
          border: '1px solid var(--border)', borderRadius: '16px',
          padding: '1.125rem 1.25rem', zIndex: 20, pointerEvents: 'auto',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
          animation: 'islandEnter 250ms ease-out, float 4s ease-in-out 250ms infinite',
        }}
      >
        <style>{`
          @keyframes islandEnter { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
          @keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-2px); } }
        `}</style>

        {/* Header */}
        <div style={{ fontFamily: 'var(--font-display)', fontSize: '0.9375rem', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
          Segment {segment.segment_id.replace('seg_', '#')}
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6875rem', color: 'var(--text-tertiary)', marginBottom: '0.125rem' }}>
          Water PFAS: {segment.predicted_water_pfas_ng_l.toFixed(1)} ng/L
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6875rem', color: 'var(--text-tertiary)', marginBottom: '0.125rem' }}>
          Confidence: {Math.round(segment.prediction_confidence * 100)}% | Stream order: {segment.stream_order}
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6875rem', color: 'var(--text-tertiary)', marginBottom: '0.875rem' }}>
          {segment.latitude.toFixed(4)}, {segment.longitude.toFixed(4)}
        </div>

        <div style={{ height: '1px', background: 'var(--border)', marginBottom: '0.75rem' }} />

        {/* Species list */}
        {sorted.slice(0, 5).map((species) => (
          <div key={species.common_name} style={{ marginBottom: '0.625rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ width: '7px', height: '7px', borderRadius: '50%', background: STATUS_COLORS[species.safety_status_recreational], flexShrink: 0 }} />
                <span style={{ fontFamily: 'var(--font-body)', fontSize: '0.8125rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                  {species.common_name}
                </span>
              </div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem', color: 'var(--text-primary)' }}>
                {species.tissue_total_pfas_ng_g.toFixed(1)} ng/g
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '0.125rem', paddingLeft: '1rem' }}>
              <span style={{ fontSize: '0.6875rem', color: 'var(--text-secondary)' }}>
                {species.safety_status_recreational === 'safe'
                  ? 'Safe for regular consumption'
                  : `Max ${species.safe_servings_per_month_recreational} serving${species.safe_servings_per_month_recreational !== 1 ? 's' : ''}/month`}
              </span>
              <button
                onClick={() => onSpeciesClick(species, segment.segment_id)}
                style={{ background: 'none', border: 'none', color: 'var(--accent)', fontSize: '0.6875rem', cursor: 'pointer', padding: '2px 0', fontFamily: 'var(--font-body)' }}
              >
                Details
              </button>
            </div>
          </div>
        ))}
        {sorted.length > 5 && (
          <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', textAlign: 'center' }}>
            +{sorted.length - 5} more species
          </div>
        )}

        {/* Demographics */}
        {nearestDemo && (
          <>
            <div style={{ height: '1px', background: 'var(--border)', margin: '0.5rem 0 0.625rem' }} />
            <div style={{ fontSize: '0.6875rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Near {nearestDemo.name} — median income ${nearestDemo.median_income.toLocaleString()}, {nearestDemo.subsistence_pct}% subsistence fishing
            </div>
          </>
        )}
      </div>
    </>
  )
}
