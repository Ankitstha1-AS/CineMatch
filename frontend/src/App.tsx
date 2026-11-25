import { useEffect, useState } from 'react'
import { getHealth } from './api/client'

type ApiStatus = 'checking' | 'ok' | 'unreachable'

export default function App() {
  const [status, setStatus] = useState<ApiStatus>('checking')
  const [version, setVersion] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    getHealth()
      .then((health) => {
        if (!cancelled) {
          setStatus('ok')
          setVersion(health.version)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setStatus('unreachable')
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  return (
    <main className="app">
      <p className="eyebrow">Final Year Project</p>
      <h1>CineMatch</h1>
      <p className="tagline">Explainable hybrid movie recommendations.</p>

      <section className="card" aria-live="polite">
        <h2>System status</h2>
        <p className={`status status--${status}`}>
          API: {status}
          {version ? ` (v${version})` : null}
        </p>
      </section>

      <footer className="footer">
        <p>Phase 0 skeleton. Recommendations arrive in the next phases.</p>
      </footer>
    </main>
  )
}
