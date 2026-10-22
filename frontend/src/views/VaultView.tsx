import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Trash2, ChevronDown, ChevronRight, Loader2, AlertTriangle, CheckCircle2, UploadCloud } from 'lucide-react'
import { documentsApi } from '../api/client'
import type { DocumentItem } from '../types'

const STATUS_STYLE: Record<string, string> = {
  done: 'text-[#00ff88] border-[#00ff88]/40 bg-[#00ff88]/10',
  failed: 'text-[#ff6e84] border-[#ff1744]/40 bg-[#ff1744]/10',
}
const KIND_STYLE: Record<string, string> = {
  prose: 'text-zinc-400 border-white/10',
  table: 'text-[#ffb703] border-[#ffb703]/40',
  caption: 'text-[#a855f7] border-[#a855f7]/40',
  footnote: 'text-[#fb7185] border-[#fb7185]/40',
  code: 'text-[#00ff88] border-[#00ff88]/40',
}

function ChunkInspector({ doc }: { doc: DocumentItem }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['chunks', doc.id],
    queryFn: () => documentsApi.getChunks(doc.id),
  })
  if (isLoading) return <div className="p-4 text-sm text-zinc-400 flex gap-2"><Loader2 className="w-4 h-4 animate-spin" /> Loading chunks…</div>
  if (error || !data) return <div className="p-4 text-sm text-[#ff8aa1]">Could not load chunks.</div>
  const stats = data.stats?.chunking
  const warnings: string[] = data.stats?.warnings ?? []
  return (
    <div className="border-t border-white/5 p-4 space-y-3">
      {stats && (
        <div className="text-[11px] font-mono text-zinc-400 flex flex-wrap gap-x-5 gap-y-1">
          <span>{stats.chunks} chunks</span>
          <span>tokens {stats.tokens_min}–{stats.tokens_max} (mean {stats.tokens_mean})</span>
          <span>{stats.skipped_reference_blocks} reference blocks excluded</span>
          <span>{stats.dropped_junk} junk blocks dropped</span>
        </div>
      )}
      {warnings.length > 0 && (
        <details className="text-xs text-[#ffcc66] bg-[#ffaa00]/10 border border-[#ffaa00]/30 rounded-lg p-3" open={warnings.length <= 3}>
          <summary className="cursor-pointer flex items-center gap-2">
            <AlertTriangle className="w-3.5 h-3.5" /> {warnings.length} parser warning(s) — content that could not be indexed
          </summary>
          <ul className="mt-2 list-disc pl-5 space-y-0.5 text-zinc-300">
            {warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </details>
      )}
      <div className="max-h-[420px] overflow-y-auto space-y-2 pr-1">
        {data.chunks.map((c) => (
          <div key={c.id} className="rounded-lg bg-black/20 border border-white/5 p-3">
            <div className="flex items-center gap-2 text-[10px] font-mono text-zinc-500 mb-1">
              <span>#{c.chunk_index}</span>
              <span className={`px-1.5 rounded border ${KIND_STYLE[c.kind] ?? ''}`}>{c.kind}</span>
              <span>p.{c.page_start}{c.page_end && c.page_end !== c.page_start ? `–${c.page_end}` : ''}</span>
              <span>{c.token_count}t</span>
              <span className="truncate text-zinc-400">{c.section_path}</span>
            </div>
            <pre className="text-[12px] leading-relaxed text-zinc-300 whitespace-pre-wrap font-sans">{c.text}</pre>
          </div>
        ))}
      </div>
    </div>
  )
}

export function VaultView({ docs, onUploadClick }: { docs: DocumentItem[]; onUploadClick: () => void }) {
  const qc = useQueryClient()
  const [open, setOpen] = useState<string | null>(null)
  const remove = useMutation({
    mutationFn: (id: string) => documentsApi.deleteDocument(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['documents'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
    },
  })

  return (
    <div className="max-w-4xl mx-auto px-6 pt-24 pb-16 h-screen overflow-y-auto">
      <div className="flex items-end justify-between mb-5">
        <div>
          <h1 className="font-headline text-2xl font-semibold text-white">Documents</h1>
          <p className="text-sm text-zinc-400">Open a document to inspect exactly how it was chunked.</p>
        </div>
        <button onClick={onUploadClick} className="px-4 py-2 rounded-full bg-gradient-to-r from-[#ff2d9b] to-[#00f0ff] text-white text-xs font-medium flex items-center gap-2 cursor-pointer">
          <UploadCloud className="w-4 h-4" /> Upload
        </button>
      </div>

      {docs.length === 0 && (
        <div className="glass-panel-subtle rounded-2xl p-10 text-center text-zinc-400 text-sm">
          Nothing indexed yet. Upload a PDF, PowerPoint, Word, Markdown, HTML or text file.
        </div>
      )}

      <div className="space-y-3">
        {docs.map((d) => {
          const busy = !['done', 'failed'].includes(d.status)
          const expanded = open === d.id
          return (
            <div key={d.id} className="glass-panel-subtle rounded-xl overflow-hidden">
              <div className="flex items-center gap-3 p-4">
                <button
                  disabled={d.status !== 'done'}
                  onClick={() => setOpen(expanded ? null : d.id)}
                  className="text-zinc-400 hover:text-white disabled:opacity-30 cursor-pointer"
                  aria-label="Toggle chunk inspector"
                >
                  {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                </button>
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium text-zinc-100 truncate">{d.title ?? d.filename}</div>
                  <div className="text-[11px] font-mono text-zinc-500 truncate">
                    {d.filename} · {d.doc_type}
                    {d.page_count ? ` · ${d.page_count} ${d.doc_type === 'pptx' ? 'slides' : 'pages'}` : ''} · {d.chunk_count} chunks
                  </div>
                  {d.status === 'failed' && d.error && <div className="mt-1 text-xs text-[#ff8aa1]">{d.error}</div>}
                </div>
                <span className={`flex items-center gap-1.5 text-[10px] font-mono uppercase px-2 py-1 rounded-full border ${STATUS_STYLE[d.status] ?? 'text-[#00f0ff] border-[#00f0ff]/40 bg-[#00f0ff]/10'}`}>
                  {busy && <Loader2 className="w-3 h-3 animate-spin" />}
                  {d.status === 'done' && <CheckCircle2 className="w-3 h-3" />}
                  {d.status}
                </span>
                <button
                  onClick={() => {
                    if (window.confirm(`Delete "${d.title ?? d.filename}" and all its chunks?`)) remove.mutate(d.id)
                  }}
                  className="text-zinc-500 hover:text-[#ff6e84] cursor-pointer"
                  aria-label="Delete document"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
              {expanded && <ChunkInspector doc={d} />}
            </div>
          )
        })}
      </div>
    </div>
  )
}
