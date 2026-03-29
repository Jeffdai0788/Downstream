import { useState, useEffect, useRef } from 'react'
import Hero from './components/Hero'
import MapView from './components/MapView'
import FloatingIsland from './components/FloatingIsland'
import DetailPanel from './components/DetailPanel'
import FishSearch from './components/FishSearch'
import AIChatPrompt from './components/AIChatPrompt'
import data from './data/nationalResults.json'

export default function App() {
  const [scrollProgress, setScrollProgress] = useState(0)
  const [hoveredSegment, setHoveredSegment] = useState(null)
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 })
  const [selectedSpecies, setSelectedSpecies] = useState(null)
  const [selectedSegment, setSelectedSegment] = useState(null)
  const [filterSpeciesName, setFilterSpeciesName] = useState(null)
  const mapRef = useRef(null)

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

  const handleSpeciesClick = (species, segmentId) => {
    setSelectedSpecies({ ...species, segmentId })
    setSelectedSegment(hoveredSegment)
    setHoveredSegment(null)
  }

  const imageOpacity = scrollProgress < 0.5 ? 1 : 1 - (scrollProgress - 0.5) / 0.5
  const mapOpacity = scrollProgress < 0.5 ? 0 : (scrollProgress - 0.5) / 0.5

  return (
    <div>
      <div style={{ height: '200vh' }} />

      <div style={{ position: 'fixed', inset: 0, zIndex: 1 }}>
        <div style={{ position: 'absolute', inset: 0, opacity: mapOpacity }}>
          <MapView
            data={data}
            onSegmentHover={handleSegmentHover}
            onCursorMove={setCursorPos}
            onMapReady={(m) => { mapRef.current = m }}
            speciesFilter={filterSpeciesName}
          />
        </div>

        <div
          style={{
            position: 'absolute', inset: 0, opacity: imageOpacity,
            pointerEvents: scrollProgress >= 0.95 ? 'none' : 'auto',
          }}
        >
          <Hero scrollProgress={scrollProgress} />
        </div>

        {scrollProgress > 0.8 && (
          <>
            {hoveredSegment && !selectedSpecies && (
              <FloatingIsland
                segment={hoveredSegment}
                demographics={data.demographics}
                position={cursorPos}
                onSpeciesClick={handleSpeciesClick}
              />
            )}

            {selectedSpecies && (
              <DetailPanel
                species={selectedSpecies}
                segment={selectedSegment}
                demographics={data.demographics}
                onClose={() => { setSelectedSpecies(null); setSelectedSegment(null) }}
              />
            )}

            <FishSearch
              data={data}
              filterSpeciesName={filterSpeciesName}
              onFilterChange={setFilterSpeciesName}
            />

            <AIChatPrompt data={data} hoveredSegment={hoveredSegment} />
          </>
        )}
      </div>
    </div>
  )
}
