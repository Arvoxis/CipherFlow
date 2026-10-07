import { Activity, Binary, FileCheck2, Shield, Sparkles } from 'lucide-react'
import { useEffect, useState } from 'react'
import { get } from './api'
import { Err, Pill, Spinner } from './ui'
import Embed from './views/Embed'
import Evasion from './views/Evasion'
import Live from './views/Live'
import Provenance from './views/Provenance'
import Tokens from './views/Tokens'

const VIEWS = [
  { id: 'live', label: 'Live classification', icon: Activity, claim: null },
  { id: 'tokens', label: 'Tokenization', icon: Binary, claim: 'A' },
  { id: 'embed', label: 'Embedding map', icon: Sparkles, claim: 'B' },
  { id: 'evasion', label: 'Evasion', icon: Shield, claim: 'C' },
  { id: 'provenance', label: 'Provenance', icon: FileCheck2, claim: null },
]

export default function App() {
  const [meta, setMeta] = useState(null)
  const [error, setError] = useState(null)
  const [view, setView] = useState('live')
  const [ckpt, setCkpt] = useState(null)
  const [dataset, setDataset] = useState(null)

  useEffect(() => {
    get('meta')
      .then((m) => {
        setMeta(m)
        const pick =
          m.checkpoints.find((c) => c.name === m.default_checkpoint) ?? m.checkpoints[0]
        setCkpt(pick?.name ?? null)
        // Open on the dataset the checkpoint was actually trained on; accuracy against any
        // other dataset is meaningless and would make the demo look broken.
        setDataset(
          m.datasets.find((d) => d.name === pick?.trained_on)?.name ?? m.datasets[0]?.name ?? null,
        )
      })
      .catch(setError)
  }, [])

  if (error)
    return (
      <div className="mx-auto max-w-xl p-10 space-y-4">
        <Err error={error} />
        <p className="text-sm text-slate-400">
          The API is not answering. Start it with:
        </p>
        <code className="block rounded-md border border-edge bg-panel px-3 py-2 font-mono text-xs text-accent">
          python -m uvicorn cipherflow.api.server:app --port 8000
        </code>
      </div>
    )
  if (!meta)
    return (
      <div className="p-10">
        <Spinner label="Loading checkpoints and datasets" />
      </div>
    )
  if (!ckpt || !dataset)
    return (
      <div className="mx-auto max-w-xl space-y-4 p-10">
        <p className="text-sm text-slate-300">
          The API is up but found no {meta.checkpoints.length ? 'datasets in data_out/' : 'classifiers in artifacts/'}.
        </p>
        <code className="block rounded-md border border-edge bg-panel px-3 py-2 font-mono text-xs text-accent">
          python -m cipherflow.pipeline --quick
        </code>
      </div>
    )

  const model = meta.checkpoints.find((c) => c.name === ckpt)
  const data = meta.datasets.find((d) => d.name === dataset)
  const mismatch = !data.labels.every((l) => model.classes.includes(l))

  return (
    <div className="flex min-h-full">
      {/* -------------------------------------------------- sidebar */}
      <aside className="w-72 shrink-0 border-r border-edge bg-panel/40 p-5">
        <div className="flex items-center gap-2.5">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-accent/15 text-accent">
            <Shield size={17} />
          </div>
          <div>
            <h1 className="text-sm font-semibold leading-tight">CipherFlow</h1>
            <p className="text-[11px] text-slate-500">flow-shape classifier</p>
          </div>
        </div>

        <nav className="mt-7 space-y-1">
          {VIEWS.map((v) => {
            const Icon = v.icon
            const on = view === v.id
            return (
              <button
                key={v.id}
                onClick={() => setView(v.id)}
                className={`flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[13px] transition ${
                  on
                    ? 'bg-accent/10 text-accent'
                    : 'text-slate-400 hover:bg-white/[0.03] hover:text-slate-200'
                }`}
              >
                <Icon size={15} className="shrink-0" />
                <span className="flex-1">{v.label}</span>
                {v.claim && (
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                      on ? 'bg-accent/20' : 'bg-edge text-slate-500'
                    }`}
                  >
                    {v.claim}
                  </span>
                )}
              </button>
            )
          })}
        </nav>

        <div className="mt-7 space-y-4">
          <label className="block">
            <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">
              Classifier
            </div>
            <select
              value={ckpt}
              onChange={(e) => {
                setCkpt(e.target.value)
                const trained = meta.checkpoints.find((c) => c.name === e.target.value)?.trained_on
                if (trained && meta.datasets.some((d) => d.name === trained)) setDataset(trained)
              }}
              className="w-full rounded-md border border-edge bg-ink px-2.5 py-1.5 font-mono text-xs outline-none focus:border-accent"
            >
              {meta.checkpoints.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">
              Dataset
            </div>
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className="w-full rounded-md border border-edge bg-ink px-2.5 py-1.5 font-mono text-xs outline-none focus:border-accent"
            >
              {meta.datasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name} ({d.flows})
                </option>
              ))}
            </select>
            {model.trained_on && model.trained_on !== dataset && (
              <p className="mt-1.5 text-[11px] text-warn">
                Trained on {model.trained_on}.
              </p>
            )}
          </label>
        </div>

        <div className="mt-7 space-y-2 border-t border-edge pt-5">
          <div className="flex flex-wrap gap-1.5">
            {model.hybrid && <Pill>hybrid head</Pill>}
            {model.augment && <Pill tone="warn">evasion-augmented</Pill>}
            <Pill tone="accent">0 payload bytes</Pill>
          </div>
          <dl className="space-y-1 pt-2 font-mono text-[11px] text-slate-500">
            <div className="flex justify-between">
              <dt>parameters</dt>
              <dd className="text-slate-300">{model.params.toLocaleString()}</dd>
            </div>
            <div className="flex justify-between">
              <dt>vocabulary</dt>
              <dd className="text-slate-300">{model.vocab.toLocaleString()}</dd>
            </div>
            <div className="flex justify-between">
              <dt>bins</dt>
              <dd className="text-slate-300">
                {model.size_bins} x {model.iat_bins}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt>classes</dt>
              <dd className="text-slate-300">{model.classes.length}</dd>
            </div>
          </dl>
        </div>
      </aside>

      {/* -------------------------------------------------- main */}
      <main className="min-w-0 flex-1 p-6">
        <header className="mb-5">
          <h2 className="text-lg font-semibold tracking-tight">
            Encrypted-traffic classification from flow shape
          </h2>
          <p className="mt-0.5 text-sm text-slate-400">
            Packet size, inter-arrival time and direction only. No payload inspection, no TLS
            termination, no decryption.
          </p>
        </header>

        {mismatch ? (
          <div className="rounded-xl border border-bad/30 bg-bad/10 p-5 text-sm text-bad">
            <p className="font-semibold">Dataset and model do not match.</p>
            <p className="mt-1.5 font-mono text-xs">
              dataset classes: {data.labels.join(', ') || 'none'}
              <br />
              model classes: {model.classes.join(', ')}
            </p>
            <p className="mt-2 text-xs">
              Pick a matching dataset in the sidebar, or retrain the classifier on this one.
            </p>
          </div>
        ) : (
          <>
            {view === 'live' && <Live ckpt={ckpt} dataset={dataset} classes={model.classes} />}
            {view === 'tokens' && <Tokens ckpt={ckpt} dataset={dataset} flows={data.flows} />}
            {view === 'embed' && <Embed ckpt={ckpt} dataset={dataset} flows={data.flows} />}
            {view === 'evasion' && (
              <Evasion
                dataset={dataset}
                checkpoints={meta.checkpoints}
                defaults={meta.robustness_defaults}
              />
            )}
            {view === 'provenance' && <Provenance ckpt={ckpt} />}
          </>
        )}
      </main>
    </div>
  )
}
