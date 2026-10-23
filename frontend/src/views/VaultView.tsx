import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Trash2,
  ChevronDown,
  ChevronRight,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  UploadCloud,
  Sparkles,
  Eye,
  Search,
  BookOpen,
} from 'lucide-react'
import { documentsApi } from '../api/client'
import type { DocumentItem, IngestionJobProgress } from '../types'

const STATUS_STYLE: Record<string, string> = {
  done: 'text-[#5ec2a8] border-[#5ec2a8]/40 bg-[#5ec2a8]/10',
  failed: 'text-[#f08a86] border-[#d9534f]/40 bg-[#d9534f]/10',
}
const KIND_STYLE: Record<string, string> = {
  prose: 'text-zinc-300 border-white/15 bg-white/5',
  table: 'text-[#e5b869] border-[#e5b869]/40 bg-[#e5b869]/10',
  caption: 'text-[#7ba7bc] border-[#7ba7bc]/40 bg-[#7ba7bc]/10',
  footnote: 'text-[#e07a5f] border-[#e07a5f]/40 bg-[#e07a5f]/10',
  code: 'text-[#5ec2a8] border-[#5ec2a8]/40 bg-[#5ec2a8]/10',
}

function ChunkInspector({ doc }: { doc: DocumentItem }) {
  const [chunkFilter, setChunkFilter] = useState('')
  const [kindFilter, setKindFilter] = useState('all')

  const { data, isLoading, error } = useQuery({
    queryKey: ['chunks', doc.id],
    queryFn: () => documentsApi.getChunks(doc.id, 300, 0),
  })

  if (isLoading)
    return (
      <div className="p-4 text-xs font-mono text-zinc-400 flex gap-2">
        <Loader2 className="w-4 h-4 animate-spin text-[#e5b869]" /> Loading indexed passages…
      </div>
    )
  if (error || !data)
    return <div className="p-4 text-xs text-[#f08a86]">Could not load chunks.</div>

  const stats = data.stats?.chunking
  const warnings: string[] = data.stats?.warnings ?? []

  const filteredChunks = data.chunks.filter((c) => {
    if (kindFilter !== 'all' && c.kind !== kindFilter) return false
    if (
      chunkFilter.trim() &&
      !c.text.toLowerCase().includes(chunkFilter.trim().toLowerCase()) &&
      !(c.section_path || '').toLowerCase().includes(chunkFilter.trim().toLowerCase())
    ) {
      return false
    }
    return true
  })

  return (
    <div className="border-t border-white/10 p-4 space-y-3 bg-black/30">
      {stats && (
        <div className="text-[11px] font-mono text-zinc-400 flex flex-wrap gap-x-5 gap-y-1">
          <span className="text-[#e5b869]">{stats.chunks} chunks</span>
          <span>
            tokens {stats.tokens_min}–{stats.tokens_max} (mean {stats.tokens_mean})
          </span>
          <span>{stats.skipped_reference_blocks} reference blocks excluded</span>
          <span>{stats.dropped_junk} junk blocks dropped</span>
        </div>
      )}

      {warnings.length > 0 && (
        <details
          className="text-xs text-[#f5d799] bg-[#e5b869]/10 border border-[#e5b869]/30 rounded-xl p-3"
          open={warnings.length <= 3}
        >
          <summary className="cursor-pointer flex items-center gap-2 font-mono">
            <AlertTriangle className="w-3.5 h-3.5" /> {warnings.length} parser warning(s)
          </summary>
          <ul className="mt-2 list-disc pl-5 space-y-0.5 text-zinc-300">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </details>
      )}

      {/* Search & Kind Filter inside Chunks */}
      <div className="flex items-center gap-2 flex-wrap">
        <div className="relative flex-1 min-w-[180px]">
          <Search className="w-3.5 h-3.5 text-zinc-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={chunkFilter}
            onChange={(e) => setChunkFilter(e.target.value)}
            placeholder="Filter passages or section headings…"
            className="w-full pl-8 pr-3 py-1.5 rounded-xl bg-black/50 border border-white/10 text-xs font-mono text-zinc-200 focus:outline-none focus:border-[#e5b869]"
          />
        </div>
        {['all', 'prose', 'table', 'caption', 'code', 'footnote'].map((k) => (
          <button
            key={k}
            onClick={() => setKindFilter(k)}
            className={`px-2.5 py-1 rounded-lg text-[10px] font-mono uppercase cursor-pointer ${
              kindFilter === k
                ? 'bg-[#e5b869]/20 text-[#e5b869] border border-[#e5b869]/40'
                : 'bg-white/5 text-zinc-400 hover:text-white border border-white/10'
            }`}
          >
            {k}
          </button>
        ))}
      </div>

      <div className="max-h-[400px] overflow-y-auto space-y-2 pr-1">
        {filteredChunks.map((c) => (
          <div key={c.id} className="rounded-xl bg-black/40 border border-white/8 p-3">
            <div className="flex items-center gap-2 text-[10px] font-mono text-zinc-400 mb-1.5">
              <span className="text-[#f3efe6] font-semibold">#{c.chunk_index}</span>
              <span className={`px-1.5 py-0.2 rounded border ${KIND_STYLE[c.kind] ?? ''}`}>
                {c.kind}
              </span>
              <span>
                p.{c.page_start}
                {c.page_end && c.page_end !== c.page_start ? `–${c.page_end}` : ''}
              </span>
              <span className="text-[#e5b869]">{c.token_count} tok</span>
              <span className="truncate text-zinc-400">{c.section_path}</span>
            </div>
            <pre className="text-[12px] leading-relaxed text-zinc-200 whitespace-pre-wrap font-sans">
              {c.text}
            </pre>
          </div>
        ))}
      </div>
    </div>
  )
}

interface VaultViewProps {
  docs: DocumentItem[]
  jobStreams: Record<string, IngestionJobProgress>
  onUploadClick: () => void
  onDropFiles: (files: FileList) => void
  onFilter3DByDoc: (docId: string) => void
  onAskDoc: (docTitle: string) => void
  onSwitchToPure3D: () => void
}

export function VaultView({
  docs,
  jobStreams,
  onUploadClick,
  onDropFiles,
  onFilter3DByDoc,
  onAskDoc,
  onSwitchToPure3D,
}: VaultViewProps) {
  const qc = useQueryClient()
  const [open, setOpen] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)

  const remove = useMutation({
    mutationFn: (id: string) => documentsApi.deleteDocument(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['documents'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
      qc.invalidateQueries({ queryKey: ['graph'] })
    },
  })

  return (
    <div className="fixed top-16 bottom-5 left-1/2 -translate-x-1/2 w-[calc(100vw-2rem)] max-w-4xl z-30 overflow-y-auto px-4 pt-3 pb-8 pointer-events-auto">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          if (e.dataTransfer.files?.length) onDropFiles(e.dataTransfer.files)
        }}
        className={`glass-panel rounded-3xl p-6 border transition-all shadow-[0_20px_70px_rgba(0,0,0,0.75)] backdrop-blur-2xl ${
          dragging ? 'border-[#e5b869] bg-[#e5b869]/10' : 'border-[#e5b869]/18'
        }`}
      >
        <div className="flex items-start justify-between flex-wrap gap-4 mb-5">
          <div>
            <h1 className="font-brand text-xl font-bold text-[#f3efe6] flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-[#e5b869]" />
              <span>Knowledge Vault & Ingestion Telemetry ({docs.length})</span>
            </h1>
            <p className="text-xs text-zinc-400 mt-1">
              Drop PDF, Markdown, or text documents to chunk, embed, and extract 3D knowledge graph entities & relations.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={onUploadClick}
              className="px-4 py-2 rounded-full bg-gradient-to-r from-[#e5b869] to-[#c8963e] text-[#080a0a] text-xs font-semibold flex items-center gap-2 shadow-[0_0_20px_rgba(229,184,105,0.3)] cursor-pointer"
            >
              <UploadCloud className="w-4 h-4" />
              <span>Upload Document</span>
            </button>
            <button
              onClick={onSwitchToPure3D}
              className="px-3 py-2 rounded-full text-[11px] font-mono bg-[#e5b869]/15 hover:bg-[#e5b869]/25 border border-[#e5b869]/40 text-[#e5b869] flex items-center gap-1.5 cursor-pointer"
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Full 3D</span>
            </button>
          </div>
        </div>

        {/* Live SSE Ingestion Streams */}
        {Object.entries(jobStreams).length > 0 && (
          <div className="mb-4 space-y-2">
            {Object.entries(jobStreams).map(([jobId, prog]) => {
              const pct =
                prog.chunks_total && prog.chunks_done != null
                  ? Math.min(100, Math.round((prog.chunks_done / prog.chunks_total) * 100))
                  : 30
              return (
                <div
                  key={jobId}
                  className="p-3 rounded-2xl bg-[#e5b869]/10 border border-[#e5b869]/35 text-xs font-mono space-y-1.5"
                >
                  <div className="flex items-center justify-between text-[#e5b869]">
                    <span className="flex items-center gap-2">
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>LIVE SSE INGESTION · Stage: {prog.stage.toUpperCase()}</span>
                    </span>
                    <span>
                      {prog.chunks_done != null && prog.chunks_total != null
                        ? `${prog.chunks_done} / ${prog.chunks_total} chunks (${pct}%)`
                        : prog.status}
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-black/50 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-[#e5b869] to-[#5ec2a8] transition-all duration-300"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {docs.length === 0 && (
          <div className="rounded-2xl p-10 text-center text-zinc-400 text-xs font-mono bg-white/[0.02] border border-dashed border-white/15">
            Drop a PDF, Markdown, or text file anywhere here to start neural extraction.
          </div>
        )}

        <div className="space-y-3">
          {docs.map((d) => {
            const busy = !['done', 'failed'].includes(d.status)
            const expanded = open === d.id
            return (
              <div
                key={d.id}
                className="glass-panel-subtle rounded-2xl overflow-hidden border border-white/10"
              >
                <div className="flex items-center gap-3 p-4 flex-wrap sm:flex-nowrap">
                  <button
                    disabled={d.status !== 'done'}
                    onClick={() => setOpen(expanded ? null : d.id)}
                    className="text-zinc-400 hover:text-white disabled:opacity-30 cursor-pointer"
                    aria-label="Toggle chunk inspector"
                  >
                    {expanded ? (
                      <ChevronDown className="w-4 h-4" />
                    ) : (
                      <ChevronRight className="w-4 h-4" />
                    )}
                  </button>

                  <div
                    onClick={() => d.status === 'done' && setOpen(expanded ? null : d.id)}
                    className="min-w-0 flex-1 cursor-pointer"
                  >
                    <div className="text-sm font-semibold text-[#f3efe6] truncate">
                      {d.title ?? d.filename}
                    </div>
                    <div className="text-[11px] font-mono text-zinc-400 truncate mt-0.5">
                      {d.filename} · {d.doc_type.toUpperCase()}
                      {d.page_count ? ` · ${d.page_count} pages` : ''} ·{' '}
                      <span className="text-[#e5b869]">{d.chunk_count} chunks</span>
                      {d.token_count ? ` · ${d.token_count.toLocaleString()} tokens` : ''}
                    </div>
                    {d.status === 'failed' && d.error && (
                      <div className="mt-1 text-xs text-[#f08a86]">{d.error}</div>
                    )}
                  </div>

                  {/* Quick 3D & RAG Actions for this Document */}
                  {d.status === 'done' && (
                    <div className="flex items-center gap-1.5">
                      <button
                        onClick={() => onFilter3DByDoc(d.id)}
                        className="px-2.5 py-1 rounded-lg bg-[#5ec2a8]/15 hover:bg-[#5ec2a8]/25 border border-[#5ec2a8]/35 text-[#5ec2a8] text-[10px] font-mono flex items-center gap-1 transition-all cursor-pointer"
                        title="Filter 3D Node Cosmos to entities originating from this paper"
                      >
                        <Eye className="w-3 h-3" />
                        <span>3D Nodes</span>
                      </button>
                      <button
                        onClick={() => onAskDoc(d.title ?? d.filename)}
                        className="px-2.5 py-1 rounded-lg bg-[#e5b869]/15 hover:bg-[#e5b869]/25 border border-[#e5b869]/35 text-[#e5b869] text-[10px] font-mono flex items-center gap-1 transition-all cursor-pointer"
                        title="Ask Ariadne specifically about this paper"
                      >
                        <Sparkles className="w-3 h-3" />
                        <span>Ask</span>
                      </button>
                    </div>
                  )}

                  <span
                    className={`flex items-center gap-1.5 text-[10px] font-mono uppercase px-2.5 py-1 rounded-full border ${
                      STATUS_STYLE[d.status] ??
                      'text-[#e5b869] border-[#e5b869]/40 bg-[#e5b869]/10'
                    }`}
                  >
                    {busy && <Loader2 className="w-3 h-3 animate-spin" />}
                    {d.status === 'done' && <CheckCircle2 className="w-3 h-3" />}
                    {d.status}
                  </span>

                  <button
                    onClick={() => {
                      if (
                        window.confirm(
                          `Delete "${d.title ?? d.filename}" and all its chunks?`
                        )
                      )
                        remove.mutate(d.id)
                    }}
                    className="p-1.5 text-zinc-500 hover:text-[#f08a86] cursor-pointer"
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
    </div>
  )
}
