import { Pause, Play, RotateCcw } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { colorFor, get, pct } from '../api'
import { Err, Panel, Range, Stat } from '../ui'

const FEED_ROWS = 14

export default function Live({ ckpt, dataset, classes }) {
  const [running, setRunning] = useState(false)
  const [offset, setOffset] = useState(0)
  const [batch, setBatch] = useState(5)
  const [speed, setSpeed] = useState(40)
  const [state, setState] = useState(null)
  const [feed, setFeed] = useState([])
  const [done, setDone] = useState(false)
  const [error, setError] = useState(null)

  // Switching model or dataset invalidates the replay; start the stream over rather than
  // continuing to append rows scored by a different checkpoint.
  useEffect(() => {
    setRunning(false)
    setOffset(0)
    setState(null)
    setFeed([])
    setDone(false)
    setError(null)
  }, [ckpt, dataset])

  useEffect(() => {
    if (!running) return
    let cancelled = false
    const timer = setTimeout(async () => {
      try {
        const d = await get('stream', { ckpt, dataset, offset, limit: batch })
        if (cancelled) return
        setState(d)
        setFeed((prev) => [...d.rows].reverse().concat(prev).slice(0, FEED_ROWS))
        if (d.done) {
          setRunning(false)
          setDone(true)
        } else setOffset((o) => o + batch)
      } catch (e) {
        if (!cancelled) {
          setError(e)
          setRunning(false)
        }
      }
    }, (1000 * batch) / speed)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [running, offset, ckpt, dataset, batch, speed])

  const reset = () => {
    setRunning(false)
    setOffset(0)
    setState(null)
    setFeed([])
    setDone(false)
    setError(null)
  }

  const counts = classes.map((c, i) => ({
    name: c,
    flows: state?.counts?.[c] ?? 0,
    fill: colorFor(i),
  }))
  const progress = state ? state.seen / state.total : 0
  const acc = state?.running_accuracy
  const held = state?.held_out_accuracy

  return (
    <div className="space-y-4">
      <Panel
        title="Live classification"
        hint="Flows stream through the model one batch at a time, exactly as they would off a network tap. Only packet size, inter-arrival time and direction reach the classifier."
        right={
          <div className="flex shrink-0 gap-2">
            <button
              onClick={() => setRunning((r) => !r)}
              disabled={done}
              className="flex items-center gap-1.5 rounded-md bg-accent px-3.5 py-1.5 text-xs font-semibold text-ink transition hover:brightness-110 disabled:opacity-40"
            >
              {running ? <Pause size={13} /> : <Play size={13} />}
              {done ? 'Finished' : running ? 'Pause' : state ? 'Resume' : 'Start'}
            </button>
            <button
              onClick={reset}
              className="flex items-center gap-1.5 rounded-md border border-edge px-3 py-1.5 text-xs text-slate-400 transition hover:border-slate-600 hover:text-slate-200"
            >
              <RotateCcw size={13} />
              Reset
            </button>
          </div>
        }
      >
        <Err error={error} />
        <div className="grid gap-5 sm:grid-cols-2">
          <Range
            label="Replay speed"
            value={speed}
            onChange={setSpeed}
            min={1}
            max={200}
            format={(v) => `${v} flows/s`}
          />
          <Range
            label="Batch per tick"
            value={batch}
            onChange={setBatch}
            min={1}
            max={50}
            format={(v) => `${v} flows`}
          />
        </div>

        <div className="mt-5 grid gap-3 sm:grid-cols-4">
          <Stat
            label="Accuracy, all flows"
            value={pct(acc)}
            tone={acc === null || acc === undefined ? 'default' : acc >= 0.5 ? 'good' : 'warn'}
            sub="includes flows seen in training"
          />
          <Stat
            label="Accuracy, held out"
            value={pct(held)}
            tone={held === null || held === undefined ? 'default' : held >= 0.5 ? 'good' : 'warn'}
            sub={`${state?.held_out_seen ?? 0} of ${state?.held_out_total ?? '--'} test flows`}
          />
          <Stat label="Flows seen" value={state ? state.seen.toLocaleString() : '0'} sub={`of ${state?.total?.toLocaleString() ?? '--'}`} />
          <Stat label="Payload bytes read" value="0" tone="good" sub="shape only" />
        </div>

        <div className="mt-4 h-1 overflow-hidden rounded-full bg-edge">
          <div
            className="h-full bg-accent transition-all duration-200"
            style={{ width: `${progress * 100}%` }}
          />
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-5">
        <Panel title="Predictions so far" className="lg:col-span-2">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={counts} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
                <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }} stroke="#1d2635" />
                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} stroke="#1d2635" allowDecimals={false} />
                <Tooltip
                  contentStyle={{
                    background: '#0e1420',
                    border: '1px solid #1d2635',
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  cursor={{ fill: '#ffffff08' }}
                />
                <Bar dataKey="flows" radius={[4, 4, 0, 0]}>
                  {counts.map((c) => (
                    <Cell key={c.name} fill={c.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Flow feed" hint="Newest first." className="lg:col-span-3">
          <div className="h-64 overflow-hidden">
            {feed.length === 0 ? (
              <p className="py-8 text-center text-sm text-slate-500">
                Press Start to stream flows through the model.
              </p>
            ) : (
              <table className="w-full text-left font-mono text-xs">
                <thead className="text-[10px] uppercase tracking-wider text-slate-500">
                  <tr>
                    <th className="pb-2 pr-4 font-medium">#</th>
                    <th className="pb-2 pr-4 font-medium">pkts</th>
                    <th className="pb-2 pr-4 font-medium">predicted</th>
                    <th className="pb-2 pr-4 font-medium">conf</th>
                    <th className="pb-2 pr-4 font-medium">actual</th>
                    <th className="pb-2 font-medium">split</th>
                  </tr>
                </thead>
                <tbody>
                  {feed.map((r) => {
                    const hit = r.actual === null || r.actual === r.prediction
                    return (
                      <tr key={r.flow} className="rowin border-t border-edge/60">
                        <td className="py-1 pr-4 text-slate-500">{r.flow}</td>
                        <td className="py-1 pr-4 text-slate-500">{r.packets}</td>
                        <td className="py-1 pr-4">
                          <span style={{ color: colorFor(classes.indexOf(r.prediction)) }}>
                            {r.prediction}
                          </span>
                        </td>
                        <td className="py-1 pr-4 text-slate-400">{r.confidence.toFixed(2)}</td>
                        <td className={`py-1 pr-4 ${hit ? 'text-slate-500' : 'text-bad'}`}>
                          {r.actual ?? '--'}
                          {!hit && ' x'}
                        </td>
                        <td className={`py-1 ${r.held_out ? 'text-accent' : 'text-slate-600'}`}>
                          {r.held_out ? 'test' : 'train'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}
          </div>
        </Panel>
      </div>
    </div>
  )
}
