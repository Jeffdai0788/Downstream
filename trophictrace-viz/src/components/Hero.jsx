const HERO_IMAGE_URL =
  'https://images.unsplash.com/photo-1497290756760-23ac55edf36f?w=1920&q=80&fit=crop'

export default function Hero({ scrollProgress }) {
  // Text fades and scrolls up during phase 1 (0–0.5)
  const textProgress = Math.min(scrollProgress / 0.5, 1)
  const textOpacity = 1 - textProgress
  const textTranslateY = textProgress * -80

  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      {/* Background image — always full opacity, parent handles fade */}
      <img
        src={HERO_IMAGE_URL}
        alt=""
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          display: 'block',
        }}
      />

      {/* Dark gradient overlay */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background:
            'linear-gradient(to bottom, rgba(25,25,25,0.3) 0%, rgba(25,25,25,0.6) 50%, rgba(25,25,25,0.9) 100%)',
        }}
      />

      {/* Title + subtitle — scrolls up and fades */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          textAlign: 'center',
          padding: '0 clamp(2rem, 5vw, 6rem)',
          opacity: textOpacity,
          transform: `translateY(${textTranslateY}px)`,
        }}
      >
        <h1
          style={{
            fontFamily: 'var(--font-display)',
            fontSize: 'clamp(2.5rem, 5vw, 4.5rem)',
            fontWeight: 400,
            lineHeight: 1.1,
            letterSpacing: '-0.02em',
            color: 'var(--text-primary)',
            marginBottom: '1.25rem',
            maxWidth: '720px',
          }}
        >
          TrophicTrace
        </h1>
        <p
          style={{
            fontFamily: 'var(--font-body)',
            fontSize: 'clamp(1rem, 1.8vw, 1.25rem)',
            fontWeight: 400,
            lineHeight: 1.55,
            color: 'var(--text-secondary)',
            maxWidth: '580px',
          }}
        >
          Predicting PFAS contamination across aquatic food webs
          using physics-informed neural networks.
        </p>

        {/* Scroll indicator */}
        <div
          style={{
            marginTop: '3rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            color: 'var(--text-tertiary)',
            fontSize: '0.8125rem',
            fontFamily: 'var(--font-body)',
            fontWeight: 400,
            opacity: textOpacity,
          }}
        >
          <span>Scroll to explore</span>
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 5v14M19 12l-7 7-7-7" />
          </svg>
        </div>
      </div>
    </div>
  )
}
