import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Send, Loader2, AlertTriangle, FileText, ShieldCheck } from 'lucide-react'
import { queryApi } from '../api/client'
import type { QueryResponse, RetrievalSource } from '../types'

interface Props {
  initialQuestion?: string
  hasDocuments: boolean
}

/** Render answer text with [n] markers as chips that scroll to the matching source. */
function AnswerText({ text, nSources }: { text: string; nSources: number }) {
  const parts = text.split(/(\[\d+\])/g)
  return (
    <p className="text-[15px] leading-relaxed text-zinc-100 whitespace-pre-wrap">
      {parts.map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/)
        if (!m) return <span key={i}>{p}</span>
        const n = Number(m[1])
        if (n < 1 || n > nSources) return null
        return (
          <a
            key={i}
            href={`#source-${n}`}
            className="mx-0.5 px-1.5 py-0.5 rounded bg-[#00f0ff]/15 border border-[#00f0ff]/40 text-[#00f0ff] text-[11px] font-mono no-underline hover:bg-[#00f0ff]/30"
          >
            {n}
          </a>
        )
      })}
    </p>
  )
}

function SourceCard({ s, cited }: { s: RetrievalSource; cited: boolean }) {
  const pages = s.pages[0] === s.pages[1] || s.pages[1] == null ? `p.${s.pages[0]}` : `pp.${s.pages[0]}–${s.pages[1]}`
  return (
    <div id={`source-${s.n}`} className={`rounded-xl p-4 glass-panel-subtle scroll-mt-24 ${cited ? 'border-[#00f0ff]/40' : ''}`}>
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-xs font-mono text-[#00f0ff]">
            <span className="px-1.5 py-0.5 rounded bg-[#00f0ff]/15 border border-[#00f0ff]/40">{s.n}</span>
            {cited && <span className="text-[#00ff88]">cited</span>}
          </div>
          <div className="mt-1 text-sm font-medium text-zinc-100 truncate">{s.doc_title ?? 'Untitled'}</div>
          <div className="text-[11px] font-mono text-zinc-400 truncate">
            {s.section_path ? `${s.section_path} · ` : ''}{pages}
          </div>
        </div>
        <div className="shrink-0 text-right text-[10px] font-mono text-zinc-500 leading-4">
          {s.scores.rerank != null && <div>rerank {s.scores.rerank.toFixed(2)}</div>}
          <div>cos {s.scores.cosine.toFixed(3)}</div>
          <div>
            {Object.entries(s.found_by).map(([k, v]) => `${k}#${v}`).join(' ')}
          </div>
        </div>
      </div>
      <p className="text-[13px] leading-relaxed text-zinc-300 whitespace-pre-wrap">{s.text}</p>
    </div>
  )
}

export function QueryView({ initialQuestion = '', hasDocuments }: Props) {
  const [question, setQuestion] = useState(initialQuestion)
  const [rerank, setRerank] = useState(true)
  const [generate, setGenerate] = useState(true)

  const ask = useMutation<QueryResponse, Error>({
    mutationFn: () => queryApi.queryRAG({ question: question.trim(), top_k: 6, generate, rerank }),
  })

  const submit = () => {
    if (question.trim().length >= 3 && !ask.isPending) ask.mutate()
  }
  const res = ask.data

  return (
    <div className="max-w-4xl mx-auto px-6 pt-24 pb-16 h-screen overflow-y-auto">
      <h1 className="font-headline text-2xl font-semibold text-white mb-1">Ask Ariadne</h1>
      <p className="text-sm text-zinc-400 mb-5">
        Answers come only from your documents, with every claim cited to the passage it came from.
      </p>

      {!hasDocuments && (
        <div className="mb-4 p-3 rounded-lg bg-[#ffaa00]/10 border border-[#ffaa00]/30 text-sm text-[#ffcc66]">
          No documents are indexed yet. Upload one from the Documents tab first.
        </div>
      )}

      <div className="glass-panel rounded-2xl p-3 flex items-end gap-3">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
          rows={2}
          placeholder="e.g. What window size does Longformer use?"
          className="flex-1 bg-transparent resize-none outline-none text-sm text-zinc-100 placeholder:text-zinc-500 px-2 py-1"
        />
        <button
          onClick={submit}
          disabled={ask.isPending || question.trim().length < 3}
          className="px-4 py-2 rounded-xl bg-gradient-to-r from-[#ff2d9b] to-[#00f0ff] text-white text-sm font-medium flex items-center gap-2 disabled:opacity-40 cursor-pointer"
        >
          {ask.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          Ask
        </button>
      </div>

      <div className="flex items-center gap-5 mt-3 text-xs text-zinc-400">
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={generate} onChange={(e) => setGenerate(e.target.checked)} />
          Generate an answer (off = just show the best passages)
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
          Cross-encoder rerank
        </label>
      </div>

      {ask.isPending && (
        <div className="mt-8 flex items-center gap-3 text-sm text-zinc-400">
          <Loader2 className="w-4 h-4 animate-spin" />
          Searching your documents{generate ? ' and writing a cited answer (the first answer after startup can take a minute while the local model loads)' : ''}…
        </div>
      )}

      {ask.isError && (
        <div className="mt-6 p-3 rounded-lg bg-[#ff1744]/10 border border-[#ff1744]/30 text-sm text-[#ff8aa1] flex gap-2">
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <div>
            Request failed: {ask.error.message}. Is the Ariadne API running on port 8100, and is Ollama running if
            "Generate an answer" is on?
          </div>
        </div>
      )}

      {res?.llm_error && (
        <div className="mt-6 p-3 rounded-lg bg-[#ffaa00]/10 border border-[#ffaa00]/30 text-sm text-[#ffcc66] flex gap-2">
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <div>
            The best passages were found, but the language model couldn't write an answer ({res.llm_error}). The
            passages below are still valid.
          </div>
        </div>
      )}

      {res && (
        <div className="mt-8 space-y-6">
          {res.answer != null && (
            <section className="glass-panel-glow rounded-2xl p-5">
              <div className="flex items-center gap-2 mb-3 text-xs font-mono uppercase tracking-widest text-[#00f0ff]">
                <ShieldCheck className="w-4 h-4" /> Answer
              </div>
              {res.refused ? (
                <p className="text-zinc-300">{res.answer}</p>
              ) : (
                <AnswerText text={res.answer} nSources={res.sources.length} />
              )}
              {res.invalid_citations.length > 0 && (
                <p className="mt-3 text-xs text-[#ffaa00]">
                  The model referenced nonexistent sources ({res.invalid_citations.join(', ')}); those markers were removed.
                </p>
              )}
              {res.uncited_sentences.length > 0 && (
                <details className="mt-3 text-xs text-[#ffaa00]">
                  <summary className="cursor-pointer">
                    {res.uncited_sentences.length} sentence(s) carry no citation — treat as unsupported
                  </summary>
                  <ul className="mt-1 list-disc pl-5 text-zinc-400">
                    {res.uncited_sentences.map((s, i) => <li key={i}>{s}</li>)}
                  </ul>
                </details>
              )}
            </section>
          )}

          <section>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-zinc-400">
                <FileText className="w-4 h-4" /> Sources ({res.sources.length})
              </div>
              <div className="text-[10px] font-mono text-zinc-500">
                {res.timing.total_ms} ms · {res.timing.candidates} candidates ·{' '}
                {res.timing.reranked ? 'reranked' : 'not reranked'}
              </div>
            </div>
            <div className="space-y-3">
              {res.sources.map((s) => (
                <SourceCard key={s.chunk_id} s={s} cited={res.cited.map(Number).includes(s.n)} />
              ))}
              {res.sources.length === 0 && <p className="text-sm text-zinc-500">No passages matched.</p>}
            </div>
          </section>
        </div>
      )}
    </div>
  )
}
