import { useState, useEffect, useMemo } from 'react'
import { ENV_CENTERS, nearestCenter } from '../data/envCenters'

// ── Constants ────────────────────────────────────────────────────────────────

const FEATURE_LABELS = {
  nearest_pfas_facility_km:  'Proximity to PFAS facility',
  upstream_npdes_pfas_count: 'Upstream PFAS dischargers',
  low_flow_7q10_m3s:         'Low-flow conditions',
  pct_urban:                 'Urban land use',
  dissolved_organic_carbon_mgl: 'Dissolved organic carbon',
  mean_annual_flow_m3s:      'River flow rate',
  afff_site_nearby:          'Nearby AFFF site',
  wwtp_upstream:             'Upstream wastewater plant',
  pfas_industry_density:     'PFAS industry density',
  stream_order:              'Stream order',
  baseflow_index:            'Baseflow index',
  pct_agriculture:           'Agricultural land use',
  pct_impervious:            'Impervious surface',
  population_density:        'Population density',
}

const TIER = (pfas) =>
  pfas >= 500 ? 'critical' : pfas >= 200 ? 'warning' : pfas >= 50 ? 'caution' : 'safe'

const TIER_CFG = {
  critical: { label: 'CRITICAL', color: '#EF4444', bg: 'rgba(239,68,68,0.10)',  action: 'Do not consume fish. Contact state PFAS response team immediately.' },
  warning:  { label: 'WARNING',  color: '#F97316', bg: 'rgba(249,115,22,0.10)', action: 'Limit consumption to ≤1 serving/month. Check state advisories.' },
  caution:  { label: 'CAUTION',  color: '#F59E0B', bg: 'rgba(245,158,11,0.10)', action: 'Monitor advisories. Recreational anglers: 2–4 servings/month max.' },
  safe:     { label: 'SAFE',     color: '#2EB872', bg: 'rgba(46,184,114,0.10)', action: 'Currently within safe limits. Continue monitoring for changes.' },
}

const STATUS_COLOR = { safe: '#2EB872', limited: '#F59E0B', unsafe: '#EF4444' }
const CENTER_BORDER = { epa_regional: '#06B6D4', state_pfas: '#3B82F6', national_hotline: '#8B5CF6' }
const CENTER_LABEL  = { epa_regional: 'EPA Regional', state_pfas: 'State Program', national_hotline: 'National Hotline' }
const BAR_COLORS = ['#EF4444', '#F97316', '#F59E0B', '#F59E0B', '#F59E0B']

// ── Sub-components ───────────────────────────────────────────────────────────

function TabBtn({ id, active, onClick, children }) {
  return (
    <button
      onClick={() => onClick(id)}
      style={{
        flex: 1,
        padding: '0.625rem 0',
        background: 'none',
        border: 'none',
        borderBottom: active ? '2px solid var(--accent)' : '2px solid transparent',
        color: active ? 'var(--text-primary)' : 'var(--text-tertiary)',
        fontFamily: 'var(--font-body)',
        fontSize: '0.75rem',
        fontWeight: active ? 500 : 400,
        cursor: 'pointer',
        transition: 'all 0.15s',
        letterSpacing: '0.02em',
      }}
    >
      {children}
    </button>
  )
}

function SectionLabel({ children }) {
  return (
    <div style={{
      fontFamily: 'var(--font-body)',
      fontSize: '0.5625rem',
      fontWeight: 500,
      letterSpacing: '0.12em',
      textTransform: 'uppercase',
      color: 'var(--text-tertiary)',
      marginBottom: '0.5rem',
    }}>
      {children}
    </div>
  )
}

function Card({ children, style }) {
  return (
    <div style={{
      background: 'var(--bg-secondary)',
      border: '1px solid var(--border)',
      borderRadius: '10px',
      padding: '0.875rem',
      marginBottom: '0.75rem',
      ...style,
    }}>
      {children}
    </div>
  )
}

// ── Report Tab ───────────────────────────────────────────────────────────────

function ReportTab({ segment, allData }) {
  const pfas   = segment.predicted_water_pfas_ng_l ?? 0
  const tier   = TIER(pfas)
  const tc     = TIER_CFG[tier]
  const center = nearestCenter(segment.latitude, segment.longitude)

  const worstSpecies = segment.species?.length
    ? [...segment.species].sort((a, b) => b.tissue_total_pfas_ng_g - a.tissue_total_pfas_ng_g)[0]
    : null

  const sortedSpecies = [...(segment.species ?? [])].sort(
    (a, b) => b.tissue_total_pfas_ng_g - a.tissue_total_pfas_ng_g
  )

  const factors = segment.top_contributing_features?.slice(0, 5) ?? []
  const maxImp  = factors.length ? Math.max(...factors.map(f => f.importance)) : 1

  // Demographic context
  const nearestDemo = allData.demographics?.reduce((best, d) => {
    const dist = Math.hypot(d.lat - segment.latitude, d.lng - segment.longitude)
    return (!best || dist < best.dist) ? { ...d, dist } : best
  }, null)
  const showDemo = nearestDemo && nearestDemo.dist < 0.5

  return (
    <div>
      {/* Urgency banner */}
      <Card style={{ borderLeft: `3px solid ${tc.color}`, background: tc.bg, marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.375rem' }}>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: '0.6rem', fontWeight: 600,
            letterSpacing: '0.14em', color: tc.color, padding: '0.2rem 0.5rem',
            border: `1px solid ${tc.color}`, borderRadius: '4px',
          }}>
            {tc.label}
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: tc.color, fontWeight: 500 }}>
            {pfas.toFixed(0)} ng/L
          </span>
        </div>
        <p style={{ fontFamily: 'var(--font-body)', fontSize: '0.7rem', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
          {tc.action}
        </p>
      </Card>

      {/* Most affected species */}
      {worstSpecies && (
        <>
          <SectionLabel>Most Affected Species</SectionLabel>
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <div style={{ fontFamily: 'var(--font-body)', fontWeight: 500, fontSize: '0.8125rem', color: 'var(--text-primary)', marginBottom: '0.2rem' }}>
                  {worstSpecies.common_name}
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-tertiary)' }}>
                  {worstSpecies.tissue_total_pfas_ng_g?.toFixed(1)} ng/g tissue
                </div>
              </div>
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: '0.6rem', fontWeight: 600,
                color: STATUS_COLOR[worstSpecies.safety_status_recreational] ?? '#888',
                border: `1px solid ${STATUS_COLOR[worstSpecies.safety_status_recreational] ?? '#888'}`,
                padding: '0.2rem 0.45rem', borderRadius: '4px', textTransform: 'uppercase', letterSpacing: '0.08em',
              }}>
                {worstSpecies.safety_status_recreational}
              </span>
            </div>
            <div style={{ marginTop: '0.5rem', fontFamily: 'var(--font-body)', fontSize: '0.6875rem', color: 'var(--text-secondary)' }}>
              {worstSpecies.safety_status_recreational === 'unsafe'
                ? 'Do not consume'
                : `Safe: ${worstSpecies.safe_servings_per_month_recreational} servings/month (recreational)`}
            </div>
          </Card>
        </>
      )}

      {/* Contributing factors */}
      {factors.length > 0 && (
        <>
          <SectionLabel>Top Contributing Factors</SectionLabel>
          <Card style={{ paddingBottom: '0.625rem' }}>
            {factors.map((f, i) => (
              <div key={f.feature} style={{ marginBottom: i < factors.length - 1 ? '0.625rem' : 0 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                  <span style={{ fontFamily: 'var(--font-body)', fontSize: '0.6875rem', color: 'var(--text-secondary)' }}>
                    {FEATURE_LABELS[f.feature] ?? f.feature}
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.625rem', color: 'var(--text-tertiary)' }}>
                    {(f.importance * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={{ height: '4px', background: 'var(--border)', borderRadius: '2px', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', borderRadius: '2px',
                    width: `${(f.importance / maxImp) * 100}%`,
                    background: BAR_COLORS[i],
                    transition: 'width 0.4s ease',
                  }} />
                </div>
              </div>
            ))}
          </Card>
        </>
      )}

      {/* Species roster */}
      {sortedSpecies.length > 0 && (
        <>
          <SectionLabel>Species at this Location</SectionLabel>
          <Card style={{ padding: '0.5rem 0.875rem' }}>
            {sortedSpecies.map((sp, i) => (
              <div key={sp.common_name} style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '0.4rem 0',
                borderBottom: i < sortedSpecies.length - 1 ? '1px solid var(--border)' : 'none',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{
                    width: '7px', height: '7px', borderRadius: '50%', flexShrink: 0,
                    background: STATUS_COLOR[sp.safety_status_recreational] ?? '#888',
                  }} />
                  <span style={{ fontFamily: 'var(--font-body)', fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                    {sp.common_name}
                  </span>
                </div>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-tertiary)' }}>
                  {sp.tissue_total_pfas_ng_g?.toFixed(1)} ng/g
                </span>
              </div>
            ))}
          </Card>
        </>
      )}

      {/* Nearest environmental center */}
      {center && (
        <>
          <SectionLabel>Nearest Resource</SectionLabel>
          <Card style={{ borderLeft: `3px solid ${CENTER_BORDER[center.type]}` }}>
            <div style={{ fontFamily: 'var(--font-body)', fontSize: '0.75rem', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
              {center.name}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>
              {center.distance_km} km away
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <a href={`tel:${center.phone}`} style={{
                fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: CENTER_BORDER[center.type],
                textDecoration: 'none', border: `1px solid ${CENTER_BORDER[center.type]}`,
                padding: '0.2rem 0.5rem', borderRadius: '4px',
              }}>
                {center.phone}
              </a>
              <a href={center.url} target="_blank" rel="noreferrer" style={{
                fontFamily: 'var(--font-body)', fontSize: '0.65rem', color: 'var(--text-tertiary)',
                textDecoration: 'none', border: '1px solid var(--border)',
                padding: '0.2rem 0.5rem', borderRadius: '4px',
              }}>
                Website ↗
              </a>
            </div>
          </Card>
        </>
      )}

      {/* Demographic context */}
      {showDemo && (
        <>
          <SectionLabel>Nearby Community</SectionLabel>
          <Card>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: '0.6875rem', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
              Near <strong style={{ color: 'var(--text-primary)' }}>{nearestDemo.name}</strong> (median income ${nearestDemo.median_income?.toLocaleString()}){' '}
              — {nearestDemo.subsistence_pct}% of households rely on locally caught fish.
            </p>
          </Card>
        </>
      )}
    </div>
  )
}

// ── Alerts Tab ───────────────────────────────────────────────────────────────

function AlertsTab({ allData, onAlertClick }) {
  const [expanded, setExpanded] = useState(false)

  const alerts = useMemo(() => {
    return allData.segments
      .filter(seg => seg.predicted_water_pfas_ng_l >= 100 || seg.risk_level === 'high')
      .map(seg => {
        const pfas = seg.predicted_water_pfas_ng_l
        const tier = TIER(pfas)
        const worstSp = seg.species?.reduce((a, b) =>
          b.tissue_total_pfas_ng_g > a.tissue_total_pfas_ng_g ? b : a, seg.species[0])
        return { segment: seg, pfas, tier, worstSp }
      })
      .sort((a, b) => b.pfas - a.pfas)
  }, [allData])

  const shown = expanded ? alerts : alerts.slice(0, 5)

  return (
    <div>
      <div style={{ fontFamily: 'var(--font-body)', fontSize: '0.6875rem', color: 'var(--text-tertiary)', marginBottom: '0.75rem' }}>
        {alerts.length} locations above alert threshold
      </div>

      {shown.map(({ segment: seg, pfas, tier, worstSp }) => {
        const tc = TIER_CFG[tier]
        return (
          <button
            key={seg.segment_id}
            onClick={() => onAlertClick(seg)}
            style={{
              width: '100%', textAlign: 'left', cursor: 'pointer',
              background: tc.bg, border: `1px solid ${tc.color}33`,
              borderLeft: `3px solid ${tc.color}`,
              borderRadius: '8px', padding: '0.625rem 0.75rem',
              marginBottom: '0.5rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.575rem', fontWeight: 600, color: tc.color, letterSpacing: '0.1em' }}>
                {tc.label}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: tc.color, fontWeight: 500 }}>
                {pfas.toFixed(0)} ng/L
              </span>
            </div>
            <div style={{ fontFamily: 'var(--font-body)', fontSize: '0.7rem', color: 'var(--text-primary)', marginBottom: '0.2rem' }}>
              {seg.segment_id}
            </div>
            {worstSp && (
              <div style={{ fontFamily: 'var(--font-body)', fontSize: '0.625rem', color: 'var(--text-tertiary)' }}>
                {worstSp.common_name} — {worstSp.tissue_total_pfas_ng_g?.toFixed(1)} ng/g tissue
              </div>
            )}
          </button>
        )
      })}

      {alerts.length > 5 && (
        <button
          onClick={() => setExpanded(v => !v)}
          style={{
            width: '100%', padding: '0.5rem',
            background: 'none', border: '1px solid var(--border)',
            borderRadius: '8px', color: 'var(--text-tertiary)',
            fontFamily: 'var(--font-body)', fontSize: '0.7rem',
            cursor: 'pointer', marginTop: '0.25rem',
          }}
        >
          {expanded ? 'Show fewer' : `Show all ${alerts.length} alerts`}
        </button>
      )}
    </div>
  )
}

// ── Resources Tab ────────────────────────────────────────────────────────────

function ResourcesTab() {
  const groups = [
    { type: 'national_hotline', label: 'National Hotlines' },
    { type: 'state_pfas',       label: 'State PFAS Programs' },
    { type: 'epa_regional',     label: 'EPA Regional Offices' },
  ]

  return (
    <div>
      {groups.map(({ type, label }) => (
        <div key={type} style={{ marginBottom: '1rem' }}>
          <SectionLabel>{label}</SectionLabel>
          {ENV_CENTERS.filter(c => c.type === type).map(c => (
            <Card key={c.id} style={{ borderLeft: `3px solid ${CENTER_BORDER[type]}`, marginBottom: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.375rem' }}>
                <div style={{ fontFamily: 'var(--font-body)', fontSize: '0.7rem', fontWeight: 500, color: 'var(--text-primary)', lineHeight: 1.35, flex: 1, marginRight: '0.5rem' }}>
                  {c.name}
                </div>
                <span style={{
                  fontFamily: 'var(--font-mono)', fontSize: '0.5rem', letterSpacing: '0.08em',
                  color: CENTER_BORDER[type], border: `1px solid ${CENTER_BORDER[type]}33`,
                  padding: '0.15rem 0.35rem', borderRadius: '3px', whiteSpace: 'nowrap', flexShrink: 0,
                }}>
                  {CENTER_LABEL[type]}
                </span>
              </div>
              {c.states[0] !== '*' && (
                <div style={{ fontFamily: 'var(--font-body)', fontSize: '0.6rem', color: 'var(--text-tertiary)', marginBottom: '0.375rem' }}>
                  Covers: {c.states.join(', ')}
                </div>
              )}
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <a href={`tel:${c.phone}`} style={{
                  fontFamily: 'var(--font-mono)', fontSize: '0.625rem', color: CENTER_BORDER[type],
                  textDecoration: 'none', border: `1px solid ${CENTER_BORDER[type]}`,
                  padding: '0.2rem 0.5rem', borderRadius: '4px',
                }}>
                  {c.phone}
                </a>
                <a href={c.url} target="_blank" rel="noreferrer" style={{
                  fontFamily: 'var(--font-body)', fontSize: '0.625rem', color: 'var(--text-tertiary)',
                  textDecoration: 'none', border: '1px solid var(--border)',
                  padding: '0.2rem 0.5rem', borderRadius: '4px',
                }}>
                  Website ↗
                </a>
              </div>
            </Card>
          ))}
        </div>
      ))}
    </div>
  )
}

// ── Main ActionPanel ──────────────────────────────────────────────────────────

export default function ActionPanel({ segment, data, onClose, onAlertClick }) {
  const [tab, setTab] = useState('report')

  useEffect(() => { setTab('report') }, [segment?.segment_id])

  const pfas = segment?.predicted_water_pfas_ng_l ?? 0
  const tier = TIER(pfas)
  const tc   = TIER_CFG[tier]

  return (
    <div style={{
      position: 'fixed',
      top: 0, left: 0, bottom: 0,
      width: '360px',
      zIndex: 50,
      display: 'flex',
      flexDirection: 'column',
      background: 'rgba(22, 22, 22, 0.96)',
      backdropFilter: 'blur(16px)',
      borderRight: '1px solid var(--border)',
      animation: 'action-panel-in 0.25s ease-out both',
    }}>

      {/* Header */}
      <div style={{
        padding: '1rem 1rem 0',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '0.625rem' }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
              <div style={{
                width: '8px', height: '8px', borderRadius: '50%', flexShrink: 0,
                background: tc.color, boxShadow: `0 0 6px ${tc.color}`,
              }} />
              <span style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', fontWeight: 500, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
                Action Center
              </span>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-tertiary)', letterSpacing: '0.08em' }}>
              {segment?.segment_id} · {segment?.latitude?.toFixed(3)}°N {Math.abs(segment?.longitude ?? 0).toFixed(3)}°W
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none', border: '1px solid var(--border)', borderRadius: '6px',
              color: 'var(--text-tertiary)', cursor: 'pointer',
              width: '28px', height: '28px', display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.875rem', flexShrink: 0, marginLeft: '0.5rem',
            }}
          >
            ✕
          </button>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: '0' }}>
          <TabBtn id="report"    active={tab === 'report'}    onClick={setTab}>Report</TabBtn>
          <TabBtn id="alerts"    active={tab === 'alerts'}    onClick={setTab}>Alerts</TabBtn>
          <TabBtn id="resources" active={tab === 'resources'} onClick={setTab}>Resources</TabBtn>
        </div>
      </div>

      {/* Scrollable content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '0.875rem 1rem' }}>
        {tab === 'report'    && <ReportTab    segment={segment} allData={data} />}
        {tab === 'alerts'    && <AlertsTab    allData={data}    onAlertClick={onAlertClick} />}
        {tab === 'resources' && <ResourcesTab />}
      </div>
    </div>
  )
}
