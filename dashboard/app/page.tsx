'use client'

import { useEffect, useState, useCallback } from 'react'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

type Stats = {
  total_requests: number
  total_flagged: number
  flag_rate_pct: number
  avg_latency_ms: number
}

type LogEntry = {
  id: number
  timestamp: string
  flagged: boolean
  score: number
  flagged_content: string | null
  reason: string | null
  detection_method: string | null
  request_preview: string
  latency_ms: number
}

function methodStyle(method: string): string {
  if (method.startsWith('base64_')) return 'bg-pink-500/15 text-pink-400'
  if (method.startsWith('unicode_')) return 'bg-cyan-500/15 text-cyan-400'
  const MAP: Record<string, string> = {
    keyword: 'bg-orange-500/15 text-orange-400',
    embedding_similarity: 'bg-purple-500/15 text-purple-400',
    length_heuristic: 'bg-yellow-500/15 text-yellow-400',
  }
  return MAP[method] ?? 'bg-zinc-500/15 text-zinc-400'
}

function MethodBadge({ method }: { method: string }) {
  const label = method.replace(/_/g, ' ')
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${methodStyle(method)}`}>
      {label}
    </span>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 px-6 py-5 flex flex-col gap-1">
      <span className="text-xs font-medium text-zinc-500 uppercase tracking-widest">{label}</span>
      <span className="text-2xl font-semibold text-white tabular-nums">{value}</span>
    </div>
  )
}

function Field({
  label,
  value,
  mono,
  highlight,
}: {
  label: string
  value: string
  mono?: boolean
  highlight?: boolean
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-zinc-500 uppercase tracking-widest">{label}</span>
      <span
        className={`text-sm rounded-lg px-3 py-2 whitespace-pre-wrap break-all ${mono ? 'font-mono' : ''} ${
          highlight
            ? 'bg-red-500/10 text-red-300 border border-red-500/20'
            : 'bg-white/5 text-zinc-200'
        }`}
      >
        {value}
      </span>
    </div>
  )
}

function SlideOver({ log, onClose }: { log: LogEntry; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative w-full max-w-lg bg-[#111] border-l border-white/10 h-full overflow-y-auto p-8 flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">Request Detail</h2>
          <button
            onClick={onClose}
            className="text-zinc-500 hover:text-white transition-colors text-xl leading-none"
          >
            ✕
          </button>
        </div>

        <div className="flex items-center gap-3">
          {log.flagged ? (
            <span className="px-2.5 py-1 rounded-full bg-red-500/15 text-red-400 text-xs font-semibold tracking-wide">
              BLOCKED
            </span>
          ) : (
            <span className="px-2.5 py-1 rounded-full bg-emerald-500/15 text-emerald-400 text-xs font-semibold tracking-wide">
              CLEAN
            </span>
          )}
          <span className="text-zinc-500 text-sm">
            {new Date(log.timestamp).toLocaleString()}
          </span>
        </div>

        {log.detection_method && (
          <div className="flex flex-col gap-1.5">
            <span className="text-xs font-medium text-zinc-500 uppercase tracking-widest">Detection Method</span>
            <div><MethodBadge method={log.detection_method} /></div>
          </div>
        )}
        <Field label="Score" value={log.score.toFixed(4)} mono />
        <Field label="Latency" value={`${log.latency_ms} ms`} mono />
        <Field label="Request Preview" value={log.request_preview} mono />
        {log.flagged_content && (
          <Field label="Flagged Content" value={log.flagged_content} mono highlight />
        )}
        {log.reason && <Field label="Reason" value={log.reason} />}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [selected, setSelected] = useState<LogEntry | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [s, l] = await Promise.all([
        fetch(`${API}/api/stats`).then((r) => r.json()),
        fetch(`${API}/api/logs`).then((r) => r.json()),
      ])
      setStats(s)
      setLogs(l.results ?? [])
      setError(null)
      setLastUpdated(new Date())
    } catch (err) {
      console.error('Failed to refresh dashboard data', err)
      setError('Unable to reach backend API. Check NEXT_PUBLIC_API_URL and backend CORS/health.')
      // keep showing stale data if server is temporarily unreachable
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 5000)
    return () => clearInterval(id)
  }, [refresh])

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white font-sans">
      {/* Header */}
      <header className="border-b border-white/10 px-8 py-4 flex items-center gap-3">
        <span className="text-base font-semibold tracking-tight">PromptShield</span>
        <span className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
      </header>

      <main className="px-8 py-8 flex flex-col gap-8 max-w-7xl mx-auto">
        {/* Stat cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <StatCard
            label="Total Requests"
            value={stats ? stats.total_requests.toString() : '—'}
          />
          <StatCard
            label="Flagged"
            value={stats ? stats.total_flagged.toString() : '—'}
          />
          <StatCard
            label="Flag Rate"
            value={stats ? `${stats.flag_rate_pct}%` : '—'}
          />
          <StatCard
            label="Avg Latency"
            value={stats ? `${stats.avg_latency_ms} ms` : '—'}
          />
        </div>

        {/* Log table */}
        <div className="rounded-xl border border-white/10 overflow-hidden">
          <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
            <div className="flex flex-col">
              <span className="text-sm font-medium text-zinc-300">Recent Requests</span>
              {error ? (
                <span className="text-xs text-red-400">{error}</span>
              ) : (
                <span className="text-xs text-emerald-400">
                  {lastUpdated
                    ? `Connected · updated ${lastUpdated.toLocaleTimeString()}`
                    : 'Connecting...'}
                </span>
              )}
            </div>
            <span className="text-xs text-zinc-600">auto-refreshes every 5s</span>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10 text-xs text-zinc-500 uppercase tracking-wider">
                <th className="text-left px-6 py-3 font-medium">Time</th>
                <th className="text-left px-6 py-3 font-medium">Preview</th>
                <th className="text-left px-6 py-3 font-medium">Status</th>
                <th className="text-left px-6 py-3 font-medium">Method</th>
                <th className="text-left px-6 py-3 font-medium">Score</th>
                <th className="text-left px-6 py-3 font-medium">Latency</th>
              </tr>
            </thead>
            <tbody>
              {logs.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-6 py-10 text-center text-zinc-600 text-sm">
                    No requests yet
                  </td>
                </tr>
              )}
              {logs.map((log) => (
                <tr
                  key={log.id}
                  onClick={() => setSelected(log)}
                  className="border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors"
                >
                  <td className="px-6 py-3 text-zinc-500 whitespace-nowrap font-mono text-xs">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </td>
                  <td className="px-6 py-3 text-zinc-300 max-w-xs truncate font-mono text-xs">
                    {log.request_preview.slice(0, 60)}
                  </td>
                  <td className="px-6 py-3">
                    {log.flagged ? (
                      <span className="px-2 py-0.5 rounded-full bg-red-500/15 text-red-400 text-xs font-semibold">
                        BLOCKED
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 text-xs font-semibold">
                        CLEAN
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-3 text-xs">
                    {log.detection_method ? (
                      <MethodBadge method={log.detection_method} />
                    ) : (
                      <span className="text-zinc-600">—</span>
                    )}
                  </td>
                  <td className="px-6 py-3 text-zinc-400 font-mono text-xs">
                    {log.score > 0 ? log.score.toFixed(3) : '—'}
                  </td>
                  <td className="px-6 py-3 text-zinc-400 font-mono text-xs">
                    {log.latency_ms} ms
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>

      {selected && <SlideOver log={selected} onClose={() => setSelected(null)} />}
    </div>
  )
}
