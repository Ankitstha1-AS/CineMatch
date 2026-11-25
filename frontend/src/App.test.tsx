import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('App', () => {
  it('renders the title and reports a healthy API', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: 'ok', version: '0.1.0', environment: 'test' }),
      }),
    )

    render(<App />)

    expect(screen.getByRole('heading', { level: 1, name: 'CineMatch' })).toBeInTheDocument()
    expect(await screen.findByText(/API: ok/i)).toBeInTheDocument()
  })

  it('reports an unreachable API when the request fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('network down')))

    render(<App />)

    expect(await screen.findByText(/API: unreachable/i)).toBeInTheDocument()
  })
})
