// Shared chrome: the four pieces every view needs. No view-specific logic here.

export function Panel({ title, hint, right, className = '', children }) {
  return (
    <section
      className={`rounded-xl border border-edge bg-panel/70 backdrop-blur-sm ${className}`}
    >
      {(title || right) && (
        <header className="flex items-start justify-between gap-4 border-b border-edge px-5 py-3.5">
          <div>
            {title && <h2 className="text-sm font-semibold tracking-tight">{title}</h2>}
            {hint && <p className="mt-0.5 max-w-2xl text-xs leading-relaxed text-slate-400">{hint}</p>}
          </div>
          {right}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  )
}

export function Stat({ label, value, sub, tone = 'default' }) {
  const tones = {
    default: 'text-slate-100',
    good: 'text-accent',
    warn: 'text-warn',
    bad: 'text-bad',
  }
  return (
    <div className="rounded-lg border border-edge bg-ink/50 px-4 py-3">
      <div className="text-[10px] font-medium uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`mt-1 font-mono text-xl leading-tight ${tones[tone]}`}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  )
}

export function Range({ label, value, onChange, min, max, step = 1, format = (v) => v }) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="font-mono text-xs text-accent">{format(value)}</span>
      </div>
      <input
        type="range"
        className="w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  )
}

export function Pill({ children, tone = 'slate' }) {
  const tones = {
    slate: 'border-edge bg-ink/60 text-slate-400',
    accent: 'border-accent/30 bg-accent/10 text-accent',
    warn: 'border-warn/30 bg-warn/10 text-warn',
  }
  return (
    <span
      className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  )
}

export function Spinner({ label = 'Working' }) {
  return (
    <div className="flex items-center gap-2.5 py-8 text-sm text-slate-500">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-edge border-t-accent" />
      {label}
    </div>
  )
}

export function Err({ error }) {
  if (!error) return null
  return (
    <div className="rounded-lg border border-bad/30 bg-bad/10 px-4 py-3 font-mono text-xs text-bad">
      {String(error.message ?? error)}
    </div>
  )
}
