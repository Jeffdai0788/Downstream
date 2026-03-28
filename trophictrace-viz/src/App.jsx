import { useState, useEffect, useRef } from 'react'
import Hero from './components/Hero'
import MapView from './components/MapView'
import Tooltip from './components/Tooltip'
import DetailPanel from './components/DetailPanel'
import mockData from './data/mockData.json'

export default function App() {
  const [scrollProgress, setScrollProgress] = useState(0)
  const [hoveredSegment, setHoveredSegment] = useState(null)
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 })
  const [selectedSpecies, setSelectedSpecies] = useState(null)

  useEffect(() => {
    const handleScroll = () => {
      const progress = Math.min(window.scrollY / window.innerHeight, 1)
      setScrollProgress(progress)
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const handleSegmentHover = (segment, e) => {
    setHoveredSegment(segment)
    if (e) setCursorPos({ x: e.point.x, y: e.point.y })
  }

  const handleSpeciesClick = (species, segmentName) => {
    setSelectedSpecies({ ...species, segmentName })
    setHoveredSegment(null)
  }

  // Phase 1 (0–0.5): text scrolls up and fades out, image stays
  // Phase 2 (0.5–1.0): image fades out, map fades in
  const imageOpacity = scrollProgress < 0.5 ? 1 : 1 - (scrollProgress - 0.5) / 0.5
  const mapOpacity = scrollProgress < 0.5 ? 0 : (scrollProgress - 0.5) / 0.5

  return (
    <div>
      {/* Scroll spacer */}
      <div style={{ height: '200vh' }} />

      {/* Fixed fullscreen layer — everything composited here */}
      <div style={{ position: 'fixed', inset: 0, zIndex: 1 }}>
        {/* Map layer — always mounted, fades in */}
        <div style={{ position: 'absolute', inset: 0, opacity: mapOpacity }}>
          <MapView
            data={mockData}
            onSegmentHover={handleSegmentHover}
            onCursorMove={setCursorPos}
          />
        </div>

        {/* Hero image layer — fades out in phase 2 */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            opacity: imageOpacity,
            pointerEvents: scrollProgress >= 0.95 ? 'none' : 'auto',
          }}
        >
          <Hero scrollProgress={scrollProgress} />
        </div>

        {/* Tooltip + Detail panel — only interactive when map is visible */}
        {scrollProgress > 0.8 && (
          <>
            {hoveredSegment && !selectedSpecies && (
              <Tooltip
                segment={hoveredSegment}
                position={cursorPos}
                onSpeciesClick={handleSpeciesClick}
              />
            )}

            {selectedSpecies && (
              <DetailPanel
                species={selectedSpecies}
                onClose={() => setSelectedSpecies(null)}
              />
            )}
          </>
        )}
      </div>
    </div>
  )
}
