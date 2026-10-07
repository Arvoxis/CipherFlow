import { ArrowRight } from 'lucide-react'
import { useEffect, useState } from 'react'
import { get } from '../api'
import { Err, Panel, Pill, Spinner, Stat } from '../ui'

export default function Tokens({ ckpt, dataset, flows }) {
  const [idx, setIdx] = useState(0)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => setIdx(0), [dataset])

  useEffect(() => {
    let cancelled = false
    setError(null)
    get('tokenize', { ckpt, dataset, idx })
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(e))
    return () => {
      cancelled = true
    }
  }, [ckpt, dataset, idx])

  return (
    <div className="space-y-4">
      <Panel
        title="Flow-shape tokenization"
        hint="Every packet becomes one token: quantized size, quantized inter-arrival time, direction. No payload bytes are read, which is why this keeps working under TLS 1.3 and QUIC where deep packet inspection is blind."
        right={<Pill tone="accent">Claim A</Pill>}
      >
        <Err error={error} />
        <div className="flex flex-wrap items-end gap-4">
          <label className="block">
            <div className="mb-1.5 text-xs text-slate-400">Flow index</div>
            <input
              type="number"
              min={0}
              max={Math.max(0, flows - 1)}
              value={idx}
              onChange={(e) => setIdx(Math.min(Math.max(0, Number(e.target.value) || 0), flows - 1))}
              className="w-28 rounded-md border border-edge bg-ink px-3 py-1.5 font-mono text-sm outline-none focus:border-accent"
            />
          </label>
          <div className="text-xs text-slate-500">of {flows.toLocaleString()} flows</div>
          {data?.label && (
            <div className="text-sm">
              <span className="text-slate-500">true label </span>
              <span className="font-mono text-accent">{data.label}</span>
            </div>
          )}
        </div>

        {data && (
          <div className="mt-5 grid gap-3 sm:grid-cols-4">
            <Stat label="Packets in flow" value={data.packets} />
            <Stat label="Composite vocabulary" value={data.vocab.toLocaleString()} />
            <Stat label="Size x IAT bins" value={`${data.size_bins} x ${data.iat_bins}`} />
            <Stat label="Payload bytes read" value="0" tone="good" />
          </div>
        )}
      </Panel>

      <Panel title="Packet to token" hint="The left half is what the wire shows. The right half is what the model sees.">
        {!data ? (
          <Spinner label="Tokenizing" />
        ) : (
          <div className="max-h-[26rem] overflow-y-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead className="sticky top-0 bg-panel text-[10px] uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="py-2 pr-4 font-medium">pkt</th>
                  <th className="py-2 pr-4 font-medium">size B</th>
                  <th className="py-2 pr-4 font-medium">iat ms</th>
                  <th className="py-2 pr-4 font-medium">dir</th>
                  <th className="py-2 pr-4 font-medium" />
                  <th className="py-2 pr-4 font-medium">size bin</th>
                  <th className="py-2 pr-4 font-medium">iat bin</th>
                  <th className="py-2 font-medium text-accent">token id</th>
                  <th className="py-2 pr-4 font-medium">covers</th>
                </tr>
              </thead>
              <tbody>
                {data.tokens.map((t) => (
                  <tr key={t.pkt} className="border-t border-edge/60">
                    <td className="py-1.5 pr-4 text-slate-500">{t.pkt}</td>
                    <td className="py-1.5 pr-4">{t.size_B}</td>
                    <td className="py-1.5 pr-4">{t.iat_ms}</td>
                    <td className={`py-1.5 pr-4 ${t.dir === 'up' ? 'text-sky-400' : 'text-fuchsia-400'}`}>
                      {t.dir}
                    </td>
                    <td className="py-1.5 pr-4 text-slate-600">
                      <ArrowRight size={12} />
                    </td>
                    <td className="py-1.5 pr-4 text-slate-400">{t.size_bin}</td>
                    <td className="py-1.5 pr-4 text-slate-400">{t.iat_bin}</td>
                    <td className="py-1.5 pr-4 text-accent">{t.token_id}</td>
                    <td className="py-1.5 pr-4 text-slate-500">{t.token_range}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  )
}
