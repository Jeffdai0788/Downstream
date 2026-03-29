import { useEffect, useRef } from 'react'
import { X } from 'lucide-react'

const STATUS_COLORS = {
  safe: '#2EB872',
  limited: '#E0A030',
  unsafe: '#DC4444',
}

const FEATURE_LABELS = {
  nearest_pfas_facility_km: 'Proximity to PFAS facility',
  upstream_npdes_pfas_count: 'Upstream PFAS dischargers',
  low_flow_7q10_m3s: 'Low-flow conditions',
  pct_urban: 'Urban land use',
  dissolved_organic_carbon_mgl: 'Dissolved organic carbon',
  mean_annual_flow_m3s: 'River flow rate',
  afff_site_nearby: 'Nearby AFFF site',
  wwtp_upstream: 'Upstream wastewater plant',
  pfas_industry_density: 'PFAS industry density',
  stream_order: 'Stream order',
  baseflow_index: 'Baseflow index',
  pct_agriculture: 'Agricultural land use',
  pct_impervious: 'Impervious surface',
  population_density: 'Population density',
}

const FEATURE_TYPES = {
  nearest_pfas_facility_km: 'source',
  upstream_npdes_pfas_count: 'source',
  afff_site_nearby: 'source',
  wwtp_upstream: 'source',
  pfas_industry_density: 'source',
  low_flow_7q10_m3s: 'hydrologic',
  mean_annual_flow_m3s: 'hydrologic',
  stream_order: 'hydrologic',
  baseflow_index: 'hydrologic',
  pct_urban: 'environmental',
  pct_agriculture: 'environmental',
  pct_impervious: 'environmental',
  dissolved_organic_carbon_mgl: 'environmental',
  population_density: 'environmental',
}

const FACTOR_COLORS = {
  source: '#E8845A',
  ecological: '#5B8FD4',
  environmental: '#3DA89A',
  biological: '#9A6DD4',
  hydrologic: '#7A7A7A',
}

function findNearestDemographic(segment, demographics) {
  if (!demographics || demographics.length === 0) return null
  let nearest = null
  let minDist = Infinity
  demographics.forEach((d) => {
    const dist = Math.sqrt((segment.latitude - d.lat) ** 2 + (segment.longitude - d.lng) ** 2)
    if (dist < minDist && dist < 0.5) { minDist = dist; nearest = d }
  })
  return nearest
}

export default function DetailPanel({ species, segment, demographics, onClose }) {
  const panelRef = useRef(null)

  useEffect(() => {
    if (panelRef.current) {
      panelRef.current.style.transform = 'translateX(100%)'
      requestAnimationFrame(() => { panelRef.current.style.transform = 'translateX(0)' })
    }
  }, [])

  const statusColor = STATUS_COLORS[species.safety_status_recreational]
  const tissueConc = species.tissue_total_pfas_ng_g
  const epaLimit = 6
  const multiplier = (tissueConc / epaLimit).toFixed(1)
  const isOver = tissueConc > epaLimit
  const nearestDemo = segment ? findNearestDemographic(segment, demographics) : null

  return (
    <>
      <div onClick={onClose} style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.4)', zIndex: 25, cursor: 'pointer' }} />

      <div ref={panelRef} style={{
        position: 'absolute', top: 0, right: 0, width: '420px', height: '100%',
        background: 'var(--bg-primary)', borderLeft: '1px solid var(--border)',
        zIndex: 30, overflowY: 'auto', transition: 'transform 300ms ease-out', padding: '2rem',
      }}>
        <button onClick={onClose} style={{ position: 'absolute', top: '1.25rem', right: '1.25rem', background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', padding: '4px' }}>
          <X size={18} strokeWidth={1.5} />
        </button>

        {/* Header */}
        <div style={{ marginBottom: '2.5rem' }}>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.375rem', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
            {species.common_name}
          </h2>
          <p style={{ fontSize: '0.8125rem', color: 'var(--text-tertiary)', fontStyle: 'italic', marginBottom: '0.125rem' }}>
            {species.scientific_name}
          </p>
          <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', marginBottom: '2rem' }}>
            Segment {species.segmentId?.replace('seg_', '#')}
          </p>

          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '3rem', fontWeight: 500, color: statusColor, lineHeight: 1.1, marginBottom: '0.375rem' }}>
            {tissueConc.toFixed(1)}
            <span style={{ fontSize: '1rem', color: 'var(--text-secondary)', marginLeft: '0.5rem', fontWeight: 400 }}>ng/g</span>
          </div>

          <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', marginBottom: '0.75rem' }}>
            EPA screening level: {epaLimit} ng/g
          </div>

          <div style={{
            display: 'inline-block', padding: '0.25rem 0.625rem', borderRadius: '8px',
            fontSize: '0.75rem', fontFamily: 'var(--font-mono)', fontWeight: 500,
            color: isOver ? STATUS_COLORS.unsafe : STATUS_COLORS.safe,
            background: isOver ? 'rgba(220,68,68,0.12)' : 'rgba(46,184,114,0.12)',
            marginBottom: '0.5rem',
          }}>
            {multiplier}x {isOver ? 'over' : 'under'} limit
          </div>

          <div style={{ marginTop: '0.75rem', fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
            {species.safety_status_recreational === 'safe'
              ? 'Safe for regular consumption'
              : `Maximum ${species.safe_servings_per_month_recreational} serving${species.safe_servings_per_month_recreational !== 1 ? 's' : ''} per month (recreational)`}
          </div>
        </div>

        {/* Congener breakdown */}
        {species.tissue_by_congener && (
          <Section title="PFAS congener breakdown">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
              {Object.entries(species.tissue_by_congener)
                .sort(([, a], [, b]) => b - a)
                .map(([congener, value]) => (
                  <div key={congener} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>{congener}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>{value.toFixed(1)} ng/g</span>
                  </div>
                ))}
            </div>
          </Section>
        )}

        {/* Contributing factors */}
        {segment?.top_contributing_features && (
          <Section title="Why is this location contaminated?">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {segment.top_contributing_features.map((feat) => {
                const label = FEATURE_LABELS[feat.feature] || feat.feature
                const type = FEATURE_TYPES[feat.feature] || 'environmental'
                const pct = Math.round(feat.importance * 100)
                return (
                  <div key={feat.feature}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem', fontSize: '0.75rem' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>{pct}%</span>
                    </div>
                    <div style={{ height: '4px', borderRadius: '2px', background: 'var(--bg-secondary)', overflow: 'hidden' }}>
                      <div style={{ width: `${pct}%`, height: '100%', borderRadius: '2px', background: FACTOR_COLORS[type], transition: 'width 400ms ease-out' }} />
                    </div>
                  </div>
                )
              })}
            </div>
          </Section>
        )}

        {/* Contamination pathway */}
        {species.pathway && (
          <Section title="Contamination pathway">
            <Pathway pathway={species.pathway} />
          </Section>
        )}

        {/* Who is at risk */}
        <Section title="Who is at risk?">
          <ExposureDisparity species={species} demographic={nearestDemo} />
        </Section>
      </div>
    </>
  )
}

function Section({ title, children }) {
  return (
    <div style={{ marginBottom: '2rem' }}>
      <div style={{ height: '1px', background: 'var(--border)', marginBottom: '1rem' }} />
      <h3 style={{ fontFamily: 'var(--font-body)', fontSize: '0.6875rem', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-tertiary)', marginBottom: '1rem' }}>
        {title}
      </h3>
      {children}
    </div>
  )
}

function Pathway({ pathway }) {
  const steps = [
    { label: pathway.source_facility || 'Source', value: `${pathway.water_concentration_ng_l.toFixed(1)} ng/L`, annotation: pathway.source_distance_km ? `${pathway.source_distance_km} km away` : null, color: 'var(--text-tertiary)' },
    { label: 'After dilution', value: `÷${pathway.dilution_factor.toFixed(1)}`, annotation: null, color: STATUS_COLORS.limited },
    { label: 'Fish tissue', value: `${pathway.tissue_concentration_ng_g.toFixed(1)} ng/g`, annotation: `×${pathway.bcf_applied} BCF, ×${pathway.tmf_applied} TMF`, color: STATUS_COLORS.unsafe },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {steps.map((step) => (
        <div key={step.label}>
          {step.annotation && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.25rem 0 0.25rem 1.25rem' }}>
              <div style={{ width: '1px', height: '16px', background: 'var(--border)' }} />
              <span style={{ fontSize: '0.6875rem', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>{step.annotation}</span>
            </div>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.5rem 0.75rem', background: 'var(--bg-secondary)', borderRadius: '8px', borderLeft: `3px solid ${step.color}` }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.125rem' }}>{step.label}</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem', fontWeight: 500, color: step.color }}>{step.value}</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

function ExposureDisparity({ species, demographic }) {
  const recHQ = species.hazard_quotient_recreational
  const subHQ = species.hazard_quotient_subsistence
  const maxHQ = Math.max(recHQ, subHQ, 1.5)
  const thresholdPct = Math.min((1.0 / maxHQ) * 100, 100)

  const barColor = (hq) => (hq >= 1.0 ? STATUS_COLORS.unsafe : hq >= 0.5 ? STATUS_COLORS.limited : STATUS_COLORS.safe)

  const formatHQ = (hq) => hq >= 100 ? `${Math.round(hq)}` : hq.toFixed(1)

  return (
    <div>
      {[
        { label: 'Recreational angler', hq: recHQ },
        { label: 'Subsistence fisher', hq: subHQ },
      ].map(({ label, hq }) => (
        <div key={label} style={{ marginBottom: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '0.375rem' }}>
            <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>HQ {formatHQ(hq)}</span>
          </div>
          <div style={{ position: 'relative', height: '8px', borderRadius: '4px', background: 'var(--bg-secondary)' }}>
            <div style={{ width: `${Math.min((hq / maxHQ) * 100, 100)}%`, height: '100%', borderRadius: '4px', background: barColor(hq), transition: 'width 400ms ease-out' }} />
            <div style={{ position: 'absolute', left: `${thresholdPct}%`, top: '-4px', bottom: '-4px', borderLeft: '1.5px dashed var(--text-tertiary)' }} />
          </div>
        </div>
      ))}

      <div style={{ fontSize: '0.625rem', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)', textAlign: 'center', marginBottom: '1rem' }}>
        Dashed line = EPA Safety Threshold (HQ 1.0)
      </div>

      {demographic && (
        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          Near {demographic.name} (median income ${demographic.median_income.toLocaleString()}), {demographic.subsistence_pct}% of households rely on locally caught fish.
        </div>
      )}
    </div>
  )
}
