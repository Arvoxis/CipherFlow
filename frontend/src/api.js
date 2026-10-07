// Every call goes through Vite's /api proxy to the FastAPI service (see vite.config.js).

export async function get(path, params = {}) {
  const qs = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null),
  )
  const res = await fetch(`/api/${path}?${qs}`)
  if (!res.ok) {
    // FastAPI puts the useful part in {detail}; anything else is a bare status.
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `${path} failed (${res.status})`)
  }
  return res.json()
}

export const CLASS_COLORS = [
  '#4ade80',
  '#60a5fa',
  '#fbbf24',
  '#f472b6',
  '#a78bfa',
  '#2dd4bf',
  '#fb923c',
  '#e879f9',
]

export const colorFor = (i) => CLASS_COLORS[i % CLASS_COLORS.length]
export const pct = (v, d = 1) => (v === null || v === undefined ? '--' : `${(v * 100).toFixed(d)}%`)
