import { useState, useMemo } from 'react'
import { Search, X } from 'lucide-react'

const STATUS_COLORS = {
  safe: '#2EB872',
  limited: '#E0A030',
  unsafe: '#DC4444',
}

const STATUS_PRIORITY = { unsafe: 0, limited: 1, safe: 2 }

export default function FishSearch({ data, filterSpeciesName, onFilterChange }) {
  const [expanded, setExpanded] = useState(false)

  // Deduplicate species across all segments
  const speciesList = useMemo(() => {
    const map = new Map()
    data.segments.forEach((seg) => {
      seg.species.forEach((sp) => {
        const existing = map.get(sp.common_name)
        if (!existing) {
          map.set(sp.common_name, {
            common_name: sp.common_name,
            segmentCount: 1,
            worstStatus: sp.safety_status_recreational,
          })
        } else {
          existing.segmentCount++
          if (STATUS_PRIORITY[sp.safety_status_recreational] < STATUS_PRIORITY[existing.worstStatus]) {
            existing.worstStatus = sp.safety_status_recreational
          }
        }
      })
    })
    return Array.from(map.values()).sort(
      (a, b) => STATUS_PRIORITY[a.worstStatus] - STATUS_PRIORITY[b.worstStatus]
    )
  }, [data])

  return (
    <div
      style={{
        position: 'absolute',
        right: '1.5rem',
        top: '50%',
        transform: 'translateY(-50%)',
        zIndex: 15,
      }}
    >
      <div
        style={{
          width: expanded ? '260px' : '44px',
          height: expanded ? 'auto' : '44px',
          borderRadius: expanded ? '16px' : '50%',
          background: 'var(--bg-surface)',
          border: '1px solid var(--border)',
          overflow: 'hidden',
          transition: 'width 250ms ease-out, height 250ms ease-out, border-radius 250ms ease-out',
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.3)',
        }}
      >
        {!expanded ? (
          // Collapsed: circle button
          <button
            onClick={() => setExpanded(true)}
            style={{
              width: '44px',
              height: '44px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
            }}
          >
            <Search size={18} strokeWidth={1.5} />
          </button>
        ) : (
          // Expanded: species list
          <div style={{ padding: '0.875rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <span
                style={{
                  fontFamily: 'var(--font-body)',
                  fontSize: '0.6875rem',
                  fontWeight: 500,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  color: 'var(--text-tertiary)',
                }}
              >
                Filter by Species
              </span>
              <button
                onClick={() => {
                  setExpanded(false)
                  onFilterChange(null)
                }}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-tertiary)',
                  cursor: 'pointer',
                  padding: '2px',
                }}
              >
                <X size={14} strokeWidth={1.5} />
              </button>
            </div>

            {speciesList.map((sp) => {
              const isActive = filterSpeciesName === sp.common_name
              return (
                <button
                  key={sp.common_name}
                  onClick={() => onFilterChange(isActive ? null : sp.common_name)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    width: '100%',
                    padding: '0.5rem 0.625rem',
                    marginBottom: '0.25rem',
                    background: isActive ? 'var(--accent-subtle)' : 'transparent',
                    border: 'none',
                    borderLeft: isActive ? '3px solid var(--accent)' : '3px solid transparent',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    transition: 'background 150ms ease-out',
                    textAlign: 'left',
                  }}
                >
                  <div
                    style={{
                      width: '7px',
                      height: '7px',
                      borderRadius: '50%',
                      background: STATUS_COLORS[sp.worstStatus],
                      flexShrink: 0,
                    }}
                  />
                  <span
                    style={{
                      fontFamily: 'var(--font-body)',
                      fontSize: '0.8125rem',
                      fontWeight: isActive ? 500 : 400,
                      color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                      flex: 1,
                    }}
                  >
                    {sp.common_name}
                  </span>
                  <span
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.625rem',
                      color: 'var(--text-tertiary)',
                    }}
                  >
                    {sp.segmentCount} area{sp.segmentCount > 1 ? 's' : ''}
                  </span>
                </button>
              )
            })}

            {filterSpeciesName && (
              <button
                onClick={() => onFilterChange(null)}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  marginTop: '0.5rem',
                  background: 'none',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  color: 'var(--text-secondary)',
                  fontSize: '0.75rem',
                  fontFamily: 'var(--font-body)',
                  cursor: 'pointer',
                }}
              >
                Clear filter
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
