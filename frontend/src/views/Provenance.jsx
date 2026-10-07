import { AlertTriangle } from 'lucide-react'
import { useEffect, useState } from 'react'
import { get } from '../api'
import { Err, Panel, Spinner, Stat } from '../ui'

export default function Provenance({ ckpt }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    get('provenance', { ckpt })
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(e))
    return () => {
      cancelled = true
    }
  }, [ckpt])

  if (error) return <Panel><Err error={error} /></Panel>
  if (!data) return <Panel><Spinner label="Reading the run log" /></Panel>

  const cur = data.current

  return (
    <div className="space-y-4">
      <Panel
        title="Provenance"
        hint="What exactly produced the loaded model. For a patent file this is the reduction-to-practice record: a revision, a config hash and a seed, not a claim that it worked once."
      >
        <div className="grid gap-3 sm:grid-cols-4">
          <Stat label="Commit" value={(cur.git_commit ?? 'no git').slice(0, 10)} sub={cur.git_branch ?? ''} />
          <Stat label="Config hash" value={cur.config_hash} sub="full resolved config" />
          <Stat label="Seed" value={String(cur.seed)} sub="every split and init" />
          <Stat
            label="Device"
            value={cur.device.startsWith('cuda') ? 'cuda' : 'cpu'}
            sub={cur.device.replace(/^cuda:/, '')}
          />
        </div>

        {cur.git_dirty && (
          <div className="mt-4 flex items-start gap-2.5 rounded-lg border border-warn/30 bg-warn/10 px-4 py-3 text-xs text-warn">
            <AlertTriangle size={15} className="mt-0.5 shrink-0" />
            <span>
              The working tree is dirty, so this run is not reproducible from the commit alone.
              Commit before capturing a figure you intend to cite.
            </span>
          </div>
        )}

        <div className="mt-5">
          <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">
            Loaded checkpoint
          </div>
          <code className="block rounded-md border border-edge bg-ink px-3 py-2 font-mono text-xs text-accent">
            artifacts/{ckpt}
          </code>
        </div>

        <div className="mt-5 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {Object.entries(cur.versions).map(([k, v]) => (
            <div key={k} className="flex items-baseline justify-between rounded-md border border-edge/60 px-3 py-1.5 font-mono text-[11px]">
              <span className="text-slate-500">{k}</span>
              <span className="text-slate-300">{v}</span>
            </div>
          ))}
        </div>
      </Panel>

      <Panel
        title="Run log"
        hint={`artifacts/run_manifest.json, newest last. ${data.history.length} recorded runs.`}
      >
        {data.history.length === 0 ? (
          <p className="text-sm text-slate-500">
            No runs recorded yet. The manifest is written by{' '}
            <code className="text-slate-400">python -m cipherflow.pipeline</code>.
          </p>
        ) : (
          <div className="max-h-80 overflow-y-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead className="sticky top-0 bg-panel text-[10px] uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="py-2 pr-4 font-medium">timestamp (utc)</th>
                  <th className="py-2 pr-4 font-medium">step</th>
                  <th className="py-2 pr-4 font-medium">commit</th>
                  <th className="py-2 pr-4 font-medium">config</th>
                  <th className="py-2 pr-4 font-medium">device</th>
                </tr>
              </thead>
              <tbody>
                {data.history.map((h, i) => (
                  <tr key={i} className="border-t border-edge/60">
                    <td className="py-1.5 pr-4 text-slate-400">{h.timestamp_utc.replace('T', ' ').replace('+00:00', '')}</td>
                    <td className="py-1.5 pr-4 text-accent">{h.step}</td>
                    <td className="py-1.5 pr-4 text-slate-500">{(h.git_commit ?? '--').slice(0, 8)}</td>
                    <td className="py-1.5 pr-4 text-slate-500">{h.config_hash}</td>
                    <td className="py-1.5 pr-4 text-slate-500">{h.device?.startsWith('cuda') ? 'cuda' : h.device}</td>
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
