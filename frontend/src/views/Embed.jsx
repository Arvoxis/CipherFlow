import { useEffect, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import { colorFor, get, pct } from '../api'
import { Err, Panel, Pill, Range, Spinner, Stat } from '../ui'

export default function Embed({ ckpt, dataset, flows }) {
  const [n, setN] = useState(Math.min(600, flows))
  const [data, setData] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => setN(Math.min(600, flows)), [flows])

  useEffect(() => {
    let cancelled = false
    setBusy(true)
    setError(null)
    get('embed', { ckpt, dataset, n })
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(e))
      .finally(() => !cancelled && setBusy(false))
    return () => {
      cancelled = true
    }
  }, [ckpt, dataset, n])

  const series = (data?.classes ?? []).map((c, i) => ({
    name: c,
    color: colorFor(i),
    points: data.points.filter((p) => p.label === c),
  }))

  return (
    <div className="space-y-4">
      <Panel
        title="Learned flow embeddings"
        hint="A 2-D projection of the pooled flow embeddings. The masked-flow-token objective never saw a label, so clusters that separate here mean the pretraining learned something transferable rather than memorizing classes."
        right={<Pill tone="accent">Claim B</Pill>}
      >
        <Err error={error} />
        <div className="max-w-sm">
          <Range
            label="Flows to project"
            value={n}
            onChange={setN}
            min={50}
            max={Math.min(3000, flows)}
            step={50}
            format={(v) => v.toLocaleString()}
          />
        </div>
        {data && (
          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <Stat label="Embedding dimension" value={data.dim} sub="pooled, before the head" />
            <Stat
              label="Variance in view"
              value={pct(data.explained_variance[0] + data.explained_variance[1])}
              sub="first two components"
            />
            <Stat label="Points plotted" value={data.points.length.toLocaleString()} />
          </div>
        )}
      </Panel>

      <Panel title="PCA projection">
        {busy && !data ? (
          <Spinner label="Embedding flows" />
        ) : (
          <div className="h-[28rem]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
                <CartesianGrid stroke="#1d2635" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="component 1"
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  stroke="#1d2635"
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="component 2"
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  stroke="#1d2635"
                />
                <ZAxis range={[18, 18]} />
                <Tooltip
                  contentStyle={{
                    background: '#0e1420',
                    border: '1px solid #1d2635',
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(v) => Number(v).toFixed(2)}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {series.map((s) => (
                  <Scatter key={s.name} name={s.name} data={s.points} fill={s.color} fillOpacity={0.62} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}
      </Panel>
    </div>
  )
}
