import { useState, useRef, useEffect, useCallback } from 'react'
import { Sparkles } from 'lucide-react'
import { askAI } from '../utils/openai'

export default function AIChatPrompt({ data, hoveredSegment }) {
  const [proximity, setProximity] = useState(0)
  const [isHovered, setIsHovered] = useState(false)
  const [isExpanded, setIsExpanded] = useState(false)
  const [query, setQuery] = useState('')
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)
  const buttonRef = useRef(null)
  const inputRef = useRef(null)
  const containerRef = useRef(null)

  // Track mouse proximity to button
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!buttonRef.current) return
      const rect = buttonRef.current.getBoundingClientRect()
      const centerX = rect.left + rect.width / 2
      const centerY = rect.top + rect.height / 2
      const dist = Math.sqrt((e.clientX - centerX) ** 2 + (e.clientY - centerY) ** 2)
      const prox = dist < 200 ? Math.max(0.15, 1 - dist / 200) : 0.15
      setProximity(prox)
    }
    window.addEventListener('mousemove', handleMouseMove, { passive: true })
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [])

  // Focus input when expanded
  useEffect(() => {
    if (isExpanded && inputRef.current) {
      inputRef.current.focus()
    }
  }, [isExpanded])

  // Click outside to collapse
  useEffect(() => {
    if (!isExpanded) return
    const handleClick = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsExpanded(false)
        setIsHovered(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [isExpanded])

  const buildContext = useCallback(() => {
    return {
      segments: data.segments.map((s) => ({
        name: s.name,
        water_pfas_ng_l: s.predicted_water_pfas_ng_l,
        confidence: s.prediction_confidence,
        species: s.species.map((sp) => ({
          name: sp.common_name,
          tissue_pfas_ng_g: sp.tissue_total_pfas_ng_g,
          status: sp.safety_status_recreational,
          servings_per_month: sp.safe_servings_per_month_recreational,
        })),
      })),
      hovered_segment: hoveredSegment?.name || null,
    }
  }, [data, hoveredSegment])

  const handleSubmit = async () => {
    if (!query.trim() || loading) return
    setLoading(true)
    setResponse(null)
    try {
      const result = await askAI(query.trim(), buildContext())
      setResponse(result)
    } catch (err) {
      setResponse(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleExpand = () => {
    setIsExpanded(true)
  }

  const baseOpacity = isExpanded ? 1 : proximity

  return (
    <div
      ref={containerRef}
      style={{
        position: 'absolute',
        bottom: '2rem',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 15,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '0.5rem',
      }}
    >
      {/* Response card */}
      {response && (
        <div
          style={{
            width: '420px',
            maxHeight: '240px',
            overflowY: 'auto',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border)',
            borderRadius: '12px',
            padding: '1rem',
            fontSize: '0.8125rem',
            fontFamily: 'var(--font-body)',
            color: 'var(--text-primary)',
            lineHeight: 1.6,
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.3)',
            animation: 'islandEnter 200ms ease-out',
          }}
        >
          <style>{`@keyframes islandEnter { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }`}</style>
          {response}
        </div>
      )}

      {/* Button / Search bar */}
      <div
        ref={buttonRef}
        onMouseEnter={() => {
          setIsHovered(true)
          if (!isExpanded) handleExpand()
        }}
        onMouseLeave={() => {
          if (!isExpanded) setIsHovered(false)
        }}
        style={{
          width: isExpanded ? '420px' : '48px',
          height: '48px',
          borderRadius: '24px',
          background: 'var(--bg-surface)',
          border: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          overflow: 'hidden',
          transition: 'width 250ms ease-out, opacity 200ms ease-out',
          opacity: baseOpacity,
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.3)',
          cursor: isExpanded ? 'text' : 'pointer',
        }}
      >
        {/* Icon */}
        <div
          style={{
            width: '48px',
            height: '48px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            color: loading ? 'var(--accent)' : 'var(--text-secondary)',
            transition: 'color 200ms',
          }}
        >
          <Sparkles size={18} strokeWidth={1.5} style={loading ? { animation: 'pulse 1s ease-in-out infinite' } : {}} />
          {loading && (
            <style>{`@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }`}</style>
          )}
        </div>

        {/* Input */}
        {isExpanded && (
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSubmit()
            }}
            placeholder="Ask about this watershed..."
            style={{
              flex: 1,
              background: 'none',
              border: 'none',
              outline: 'none',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-body)',
              fontSize: '0.875rem',
              paddingRight: '1rem',
            }}
          />
        )}
      </div>
    </div>
  )
}
