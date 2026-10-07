import { Swords } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { get, pct } from '../api'
import { Err, Panel, Pill, Range, Spinner, Stat } from '../ui'

export default function Evasion({ dataset, checkpoints, defaults }) {
  // Default to every checkpoint trained on the dataset in view. Putting a normally-trained
  // model next to an augmented one is the entire point of this view, so defaulting to one
  // checkpoint would hide the result it exists to show.
  const sameData = checkpoints.filter((c) => c.trained_on === dataset).map((c) => c.name)
  const [picked, setPicked] = useState(sameData)
  const [pad, setPad] = useState(defaults.pad_prob)
  const [frac, setFrac] = useState(defaults.pad_max_frac)
  const [jitter, setJitter] = useState(defaults.jitter_std_ms)
  const [n, setN] = useState(800)
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    setPicked(checkpoints.filter((c) => c.trained_on === dataset).map((c) => c.name))
    setRes(null)
  }, [dataset, checkpoints])

  const run = async () => {
    if (!picked.length) return
    setBusy(true)
    setError(null)
    try {
      setRes(
        await get('evasion', {
          dataset,
          ckpts: picked.join(','),
          pad_prob: pad,
          pad_max_frac: frac,
          jitter_std_ms: jitter,
          n,
        }),
      )
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  const rows = (res?.rows ?? []).filter((r) => !r.error)
  const failed = (res?.rows ?? []).filter((r) => r.error)
  const best = rows.length ? rows.reduce((a, b) => (a.relative_loss <= b.relative_loss ? a : b)) : null
  const chart = rows.map((r) => ({
    name: r.checkpoint.replace(/\.pt$/, ''),
    clean: r.clean_acc,
    evaded: r.evaded_acc,
  }))

  return (
    <div className="space-y-4">
      <Panel
        title="Evasion robustness"
        hint="Take the attacker's position. The encryption holds, so instead they reshape the flow: pad packets to hide their sizes, jitter the timing to smear the gaps. Turn the knobs and watch what it costs the classifier."
        right={<Pill tone="accent">Claim C</Pill>}
      >
        <Err error={error} />
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          <Range label="Padding probability" value={pad} onChange={setPad} min={0} max={1} step={0.05} format={(v) => v.toFixed(2)} />
          <Range label="Max size inflation" value={frac} onChange={setFrac} min={0} max={2} step={0.05} format={(v) => `${(v * 100).toFixed(0)}%`} />
          <Range label="Timing jitter" value={jitter} onChange={setJitter} min={0} max={100} step={1} format={(v) => `${v} ms`} />
          <Range label="Test flows to score" value={n} onChange={setN} min={100} max={4000} step={100} format={(v) => v.toLocaleString()} />
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-2">
          <span className="mr-1 text-xs text-slate-400">Compare</span>
          {checkpoints.map((c) => {
            const on = picked.includes(c.name)
            return (
              <button
                key={c.name}
                onClick={() =>
                  setPicked((p) => (on ? p.filter((x) => x !== c.name) : [...p, c.name]))
                }
                className={`rounded-md border px-2.5 py-1 font-mono text-[11px] transition ${
                  on
                    ? 'border-accent/40 bg-accent/10 text-accent'
                    : 'border-edge text-slate-500 hover:border-slate-600 hover:text-slate-300'
                }`}
              >
                {c.name}
                {c.augment && <span className="ml-1.5 text-warn">aug</span>}
              </button>
            )
          })}
          <button
            onClick={run}
            disabled={busy || !picked.length}
            className="ml-auto flex items-center gap-1.5 rounded-md bg-accent px-3.5 py-1.5 text-xs font-semibold text-ink transition hover:brightness-110 disabled:opacity-40"
          >
            <Swords size={13} />
            {busy ? 'Scoring' : 'Run attack'}
          </button>
        </div>
        {!res && !busy && (
          <p className="mt-4 text-xs text-slate-500">
            Set the attacker's knobs, then run. Both the clean and the evaded pass score the same
            held-out test split the CLI uses, so these numbers match{' '}
            <code className="text-slate-400">cipherflow.robustness.eval_robustness</code>.
          </p>
        )}
      </Panel>

      {busy && !res && <Panel><Spinner label="Scoring clean and evaded flows" /></Panel>}

      {best && (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            <Stat label="Most robust" value={best.checkpoint.replace(/\.pt$/, '')} sub={best.augment ? 'evasion-augmented' : 'standard training'} />
            <Stat label="Accuracy under attack" value={pct(best.evaded_acc)} tone="good" sub={`${best.acc_lost <= 0 ? '+' : '-'}${pct(Math.abs(best.acc_lost))} vs clean`} />
            <Stat
              label="Relative loss"
              value={pct(best.relative_loss)}
              tone={best.relative_loss > 0.15 ? 'bad' : 'good'}
              sub={`${best.n} test flows`}
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="Clean vs evaded">
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chart} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
                    <CartesianGrid stroke="#1d2635" vertical={false} />
                    <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} stroke="#1d2635" />
                    <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fill: '#64748b', fontSize: 11 }} stroke="#1d2635" />
                    <Tooltip
                      contentStyle={{ background: '#0e1420', border: '1px solid #1d2635', borderRadius: 8, fontSize: 12 }}
                      formatter={(v) => pct(v, 2)}
                      cursor={{ fill: '#ffffff08' }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="clean" name="clean" fill="#60a5fa" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="evaded" name="under attack" fill="#4ade80" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-slate-500">
                A short green bar next to a tall blue one is a classifier the attacker just
                defeated. Two bars of similar height is Claim C holding.
              </p>
            </Panel>

            <Panel title="Scores" hint="Sorted by how little the attack cost.">
              <table className="w-full text-left font-mono text-xs">
                <thead className="text-[10px] uppercase tracking-wider text-slate-500">
                  <tr>
                    <th className="pb-2 pr-4 font-medium">checkpoint</th>
                    <th className="pb-2 pr-4 font-medium">clean</th>
                    <th className="pb-2 pr-4 font-medium">evaded</th>
                    <th className="pb-2 pr-4 font-medium">lost</th>
                    <th className="pb-2 pr-4 font-medium">rel.</th>
                  </tr>
                </thead>
                <tbody>
                  {[...rows]
                    .sort((a, b) => a.relative_loss - b.relative_loss)
                    .map((r) => (
                      <tr key={r.checkpoint} className="border-t border-edge/60">
                        <td className="py-1.5 pr-4">
                          {r.checkpoint.replace(/\.pt$/, '')}
                          {r.augment && <span className="ml-1.5 text-[10px] text-warn">aug</span>}
                        </td>
                        <td className="py-1.5 pr-4 text-slate-300">{r.clean_acc.toFixed(3)}</td>
                        <td className="py-1.5 pr-4 text-slate-300">{r.evaded_acc.toFixed(3)}</td>
                        <td className={`py-1.5 pr-4 ${r.acc_lost > 0 ? 'text-bad' : 'text-accent'}`}>
                          {r.acc_lost >= 0 ? '-' : '+'}
                          {Math.abs(r.acc_lost).toFixed(3)}
                        </td>
                        <td className="py-1.5 pr-4 text-slate-400">{pct(r.relative_loss)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
              {failed.map((f) => (
                <p key={f.checkpoint} className="mt-3 font-mono text-[11px] text-warn">
                  {f.checkpoint}: {f.error}
                </p>
              ))}
            </Panel>
          </div>
        </>
      )}
    </div>
  )
}
