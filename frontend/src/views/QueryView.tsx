import { useState, useEffect, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import {
  Send,
  Loader2,
  AlertTriangle,
  FileText,
  ShieldCheck,
  Sparkles,
  Network,
  CheckCircle2,
  HelpCircle,
  XCircle,
  SlidersHorizontal,
  Eye,
  Compass,
  GitBranch,
  Layers,
} from 'lucide-react'
import { queryApi } from '../api/client'
import type {
  QueryResponse,
  RetrievalSource,
  DocumentItem,
  GraphLinkNode,
  GraphNode,
} from '../types'

interface Props {
  initialQuestion?: string
  hasDocuments: boolean
  docs: DocumentItem[]
  nodes: GraphNode[]
  onHighlightGraph: (nodeNames: string[], edgePairs: string[], label: string) => void
  onFocusNodeByName: (name: string) => void
  onSwitchToPure3D: () => void
}

/** Render answer text with [n] markers as interactive glowing chips that scroll to the matching source. */
function AnswerText({ text, nSources }: { text: string; nSources: number }) {
  const parts = text.split(/(\[\d+\])/g)
  return (
    <p className="text-[14.5px] leading-relaxed text-[#f3efe6] whitespace-pre-wrap">
      {parts.map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/)
        if (!m) return <span key={i}>{p}</span>
        const n = Number(m[1])
        if (n < 1 || n > nSources) return null
        return (
          <a
            key={i}
            href={`#source-${n}`}
            className="mx-0.5 px-1.5 py-0.5 rounded bg-[#e5b869]/15 border border-[#e5b869]/45 text-[#e5b869] text-[11px] font-mono no-underline hover:bg-[#e5b869]/30 transition-colors"
          >
            [{n}]
          </a>
        )
      })}
    </p>
  )
}

function SourceCard({ s, cited }: { s: RetrievalSource; cited: boolean }) {
  const pages =
    s.pages[0] === s.pages[1] || s.pages[1] == null
      ? `p.${s.pages[0] ?? 1}`
      : `pp.${s.pages[0]}–${s.pages[1]}`
  return (
    <div
      id={`source-${s.n}`}
      className={`rounded-2xl p-4 glass-panel-subtle scroll-mt-24 transition-all ${
        cited
          ? 'border-[#e5b869]/45 shadow-[0_0_20px_rgba(229,184,105,0.1)]'
          : 'border-white/8'
      }`}
    >
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="px-2 py-0.5 rounded-md bg-[#e5b869]/15 border border-[#e5b869]/40 text-[#e5b869] font-bold">
              [{s.n}]
            </span>
            {cited && (
              <span className="px-2 py-0.5 rounded-full bg-[#5ec2a8]/15 border border-[#5ec2a8]/30 text-[#5ec2a8] text-[10px]">
                CITED IN ANSWER
              </span>
            )}
            {s.found_by?.graph != null && (
              <span className="px-2 py-0.5 rounded-full bg-[#e07a5f]/15 border border-[#e07a5f]/35 text-[#f2a38e] text-[10px]">
                GRAPH HOP {s.found_by.graph}
              </span>
            )}
          </div>
          <div className="mt-1.5 text-sm font-semibold text-[#f3efe6] truncate">
            {s.doc_title ?? 'Untitled Document'}
          </div>
          <div className="text-[11px] font-mono text-zinc-400 truncate">
            {s.section_path ? `${s.section_path} · ` : ''}
            {pages}
          </div>
        </div>
        <div className="shrink-0 text-right text-[10px] font-mono text-zinc-400 leading-4 bg-black/30 px-2.5 py-1.5 rounded-xl border border-white/5">
          {s.scores.rerank != null && (
            <div className="text-[#e5b869]">rerank {s.scores.rerank.toFixed(2)}</div>
          )}
          <div>cos {s.scores.cosine.toFixed(3)}</div>
          <div className="text-zinc-500">
            {Object.entries(s.found_by)
              .map(([k, v]) => `${k}#${v}`)
              .join(' ')}
          </div>
        </div>
      </div>
      <p className="text-[12.5px] leading-relaxed text-zinc-300 whitespace-pre-wrap bg-black/25 p-3 rounded-xl border border-white/5">
        {s.text}
      </p>
    </div>
  )
}

const SAMPLE_QUESTIONS = [
  'How does BERT use Masked Language Modeling and Next Sentence Prediction?',
  'What connects BERT and Transformer Architecture across the papers?',
  'What are the main themes and findings across all indexed documents?',
  'How does scene-adaptive dehazing improve real-time object detection?',
]

export function QueryView({
  initialQuestion = '',
  hasDocuments,
  docs,
  onHighlightGraph,
  onFocusNodeByName,
  onSwitchToPure3D,
}: Props) {
  const [question, setQuestion] = useState(initialQuestion)
  const [useGraph, setUseGraph] = useState(true)
  const [rerank, setRerank] = useState(true)
  const [verifyClaims, setVerifyClaims] = useState(false)
  const [generalKnowledge, setGeneralKnowledge] = useState(true)
  const [generate, setGenerate] = useState(true)
  const [topK, setTopK] = useState(6)
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [linkedNodes, setLinkedNodes] = useState<GraphLinkNode[] | null>(null)
  const [compactHUD, setCompactHUD] = useState(false)

  const ask = useMutation<QueryResponse, Error, string | void>({
    mutationFn: (overrideQ) => {
      const qText = (typeof overrideQ === 'string' ? overrideQ : question).trim()
      return queryApi.queryRAG({
        question: qText,
        document_ids: selectedDocIds.length > 0 ? selectedDocIds : null,
        top_k: topK,
        generate,
        rerank,
        use_graph: useGraph,
        verify_claims: verifyClaims,
        general_knowledge: generalKnowledge,
      })
    },
    onSuccess: (data) => {
      // Synchronize the 3D glowing node universe with the query's active seeds and traversed facts!
      const names = new Set<string>()
      const pairs = new Set<string>()
      data.graph?.seeds?.forEach((s) => names.add(s.toLowerCase()))
      data.graph?.facts?.forEach((f) => {
        names.add(f.subject.toLowerCase())
        names.add(f.object.toLowerCase())
        pairs.add(`${f.subject.toLowerCase()}::${f.object.toLowerCase()}`)
      })
      if (names.size > 0) {
        onHighlightGraph(
          Array.from(names),
          Array.from(pairs),
          `RAG Active Subgraph: ${names.size} nodes · ${data.graph?.hops || 1} hop(s)`
        )
      }
    },
  })

  const probeLinks = useMutation({
    mutationFn: () =>
      queryApi.linkEntities({
        question: question.trim(),
        document_ids: selectedDocIds.length > 0 ? selectedDocIds : null,
        top_k: topK,
        rerank,
      }),
    onSuccess: (data) => {
      setLinkedNodes(data.nodes)
      if (data.nodes.length > 0) {
        onHighlightGraph(
          data.nodes.map((n) => n.name.toLowerCase()),
          [],
          `Linked Seed Entities (${data.nodes.length} nodes)`
        )
      }
    },
  })

  const submit = (overrideQ?: string) => {
    const qText = (typeof overrideQ === 'string' ? overrideQ : question).trim()
    if (qText.length >= 3 && !ask.isPending) {
      setLinkedNodes(null)
      ask.mutate(qText)
    }
  }

  const autoRanRef = useRef(false)
  useEffect(() => {
    if (initialQuestion && initialQuestion.trim().length >= 3 && !autoRanRef.current) {
      autoRanRef.current = true
      setQuestion(initialQuestion)
      ask.mutate(initialQuestion.trim())
    }
  }, [initialQuestion])

  const toggleDocFilter = (id: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    )
  }

  const res = ask.data
  const doneDocs = docs.filter((d) => d.status === 'done')

  return (
    <div
      className={`fixed top-16 bottom-5 z-30 transition-all duration-300 overflow-y-auto px-4 pt-3 pb-8 pointer-events-auto ${
        compactHUD
          ? 'left-4 w-[460px] max-w-[calc(100vw-2rem)] rounded-3xl glass-panel border border-white/15 shadow-[0_20px_70px_rgba(0,0,0,0.8)]'
          : 'left-1/2 -translate-x-1/2 w-[calc(100vw-2rem)] max-w-4xl'
      }`}
    >
      <div className="glass-panel rounded-3xl p-6 border border-[#e5b869]/18 shadow-[0_20px_70px_rgba(0,0,0,0.75)] backdrop-blur-2xl">
        {/* Header Bar */}
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-[#e5b869]" />
              <h1 className="font-brand text-xl font-bold text-[#f3efe6]">
                Ask Ariadne · Neural GraphRAG Studio
              </h1>
            </div>
            <p className="text-xs text-zinc-400 mt-1">
              Hybrid vector + BM25 retrieval fused with multi-hop 3D knowledge graph traversal and NLI semantic verification.
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={() => setCompactHUD(!compactHUD)}
              className="px-3 py-1.5 rounded-full text-[11px] font-mono bg-white/8 hover:bg-white/15 border border-white/15 text-zinc-200 flex items-center gap-1.5 transition-all cursor-pointer"
              title="Dock panel to left side so you can interact with the 3D nodes alongside your answer"
            >
              <Compass className="w-3.5 h-3.5 text-[#e5b869]" />
              <span>{compactHUD ? 'Center Panel' : 'Split 3D HUD'}</span>
            </button>
            <button
              onClick={onSwitchToPure3D}
              className="px-3 py-1.5 rounded-full text-[11px] font-mono bg-[#e5b869]/15 hover:bg-[#e5b869]/25 border border-[#e5b869]/40 text-[#e5b869] flex items-center gap-1.5 transition-all cursor-pointer"
              title="Minimize overlay to explore the highlighted 3D cosmos full-screen"
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Full 3D</span>
            </button>
          </div>
        </div>

        {!hasDocuments && (
          <div className="mb-4 p-3 rounded-xl bg-[#e5b869]/10 border border-[#e5b869]/30 text-xs text-[#f5d799]">
            No documents are indexed yet. Upload a research paper from the Vault tab first.
          </div>
        )}

        {/* Sample Prompts */}
        <div className="flex items-center gap-1.5 flex-wrap mb-3">
          <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 mr-1">
            Try:
          </span>
          {SAMPLE_QUESTIONS.map((sq) => (
            <button
              key={sq}
              onClick={() => {
                setQuestion(sq)
                submit(sq)
              }}
              className="px-2.5 py-1 rounded-full text-[11px] bg-white/[0.04] hover:bg-white/10 border border-white/10 text-zinc-300 hover:text-white transition-all truncate max-w-[260px] cursor-pointer"
            >
              {sq}
            </button>
          ))}
        </div>

        {/* Question Input Box */}
        <div className="rounded-2xl bg-black/45 border border-white/15 focus-within:border-[#e5b869]/60 p-3 flex items-end gap-3 shadow-inner">
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
            placeholder="Ask anything across your papers, entities, or multi-hop relationships…"
            className="flex-1 bg-transparent resize-none outline-none text-sm text-zinc-100 placeholder:text-zinc-500 px-2 py-1"
          />
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={() => probeLinks.mutate()}
              disabled={probeLinks.isPending || question.trim().length < 3}
              title="Preview which 3D graph entities this question links to (/api/graph/link)"
              className="px-3 py-2 rounded-xl bg-white/8 hover:bg-white/15 border border-white/15 text-zinc-200 text-xs font-mono flex items-center gap-1.5 disabled:opacity-40 transition-all cursor-pointer"
            >
              {probeLinks.isPending ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin text-[#e5b869]" />
              ) : (
                <Network className="w-3.5 h-3.5 text-[#5ec2a8]" />
              )}
              <span className="hidden sm:inline">Link 3D</span>
            </button>
            <button
              onClick={() => submit()}
              disabled={ask.isPending || question.trim().length < 3}
              className="px-4 py-2 rounded-xl bg-gradient-to-r from-[#e5b869] to-[#c8963e] text-[#080a0a] text-xs font-bold flex items-center gap-2 disabled:opacity-40 shadow-[0_0_20px_rgba(229,184,105,0.3)] hover:opacity-95 transition-all cursor-pointer"
            >
              {ask.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-3.5 h-3.5" />
              )}
              <span>Synthesize</span>
            </button>
          </div>
        </div>

        {/* Pipeline Controls Strip */}
        <div className="flex items-center justify-between flex-wrap gap-3 mt-3 text-xs text-zinc-300 font-mono">
          <div className="flex items-center flex-wrap gap-4">
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={useGraph}
                onChange={(e) => setUseGraph(e.target.checked)}
                className="accent-[#e5b869]"
              />
              <span>GraphRAG Walk</span>
            </label>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={rerank}
                onChange={(e) => setRerank(e.target.checked)}
                className="accent-[#e5b869]"
              />
              <span>Cross-Encoder Rerank</span>
            </label>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={verifyClaims}
                onChange={(e) => setVerifyClaims(e.target.checked)}
                className="accent-[#5ec2a8]"
              />
              <span>NLI Claim Check</span>
            </label>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={generate}
                onChange={(e) => setGenerate(e.target.checked)}
                className="accent-[#e07a5f]"
              />
              <span>LLM Answer</span>
            </label>
          </div>

          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-[11px] text-[#e5b869] hover:underline flex items-center gap-1 cursor-pointer"
          >
            <SlidersHorizontal className="w-3 h-3" />
            <span>
              {showAdvanced ? 'Hide Scope' : `Scope & Top-K (${topK})`}
              {selectedDocIds.length > 0 ? ` · ${selectedDocIds.length} paper(s)` : ''}
            </span>
          </button>
        </div>

        {/* Expandable Advanced Scope & Top-K Controls */}
        {showAdvanced && (
          <div className="mt-3 p-3.5 rounded-2xl bg-black/40 border border-white/10 space-y-3 text-xs font-mono">
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <div className="flex items-center gap-3">
                <span className="text-zinc-400">Top-K Passages: {topK}</span>
                <input
                  type="range"
                  min={2}
                  max={15}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-28 accent-[#e5b869] cursor-pointer"
                />
              </div>
              <label className="flex items-center gap-1.5 cursor-pointer text-zinc-300">
                <input
                  type="checkbox"
                  checked={generalKnowledge}
                  onChange={(e) => setGeneralKnowledge(e.target.checked)}
                  className="accent-[#e5b869]"
                />
                <span>Allow General Knowledge Fallback if out-of-corpus</span>
              </label>
            </div>

            {doneDocs.length > 0 && (
              <div>
                <div className="text-[10px] uppercase text-zinc-500 mb-1.5 flex justify-between">
                  <span>Filter by Specific Document(s)</span>
                  {selectedDocIds.length > 0 && (
                    <button
                      onClick={() => setSelectedDocIds([])}
                      className="text-[#e5b869] hover:underline cursor-pointer"
                    >
                      Clear filter
                    </button>
                  )}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {doneDocs.map((d) => {
                    const active = selectedDocIds.includes(d.id)
                    return (
                      <button
                        key={d.id}
                        onClick={() => toggleDocFilter(d.id)}
                        className={`px-2.5 py-1 rounded-lg text-[11px] transition-all truncate max-w-[240px] cursor-pointer ${
                          active
                            ? 'bg-[#e5b869]/20 text-[#e5b869] border border-[#e5b869]/50'
                            : 'bg-white/5 text-zinc-400 hover:text-zinc-200 border border-white/10'
                        }`}
                      >
                        {d.title ?? d.filename}
                      </button>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Entity Linker Probe Output (/api/graph/link) */}
        {linkedNodes && (
          <div className="mt-4 p-3.5 rounded-2xl bg-[#5ec2a8]/8 border border-[#5ec2a8]/30">
            <div className="text-[11px] font-mono uppercase tracking-wider text-[#5ec2a8] mb-2 flex items-center justify-between">
              <span>Linked 3D Graph Entities ({linkedNodes.length})</span>
              <span className="text-[10px] text-zinc-400">Click any node to fly in 3D</span>
            </div>
            {linkedNodes.length === 0 ? (
              <p className="text-xs text-zinc-400">No relevant graph entities linked.</p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {linkedNodes.map((ln) => (
                  <button
                    key={ln.id}
                    onClick={() => onFocusNodeByName(ln.name)}
                    className="px-2.5 py-1 rounded-xl bg-black/50 hover:bg-[#5ec2a8]/20 border border-white/15 hover:border-[#5ec2a8]/50 text-xs font-mono flex items-center gap-2 transition-all cursor-pointer"
                  >
                    <span className="text-[#f3efe6] font-semibold">{ln.name}</span>
                    <span className="text-[10px] text-[#5ec2a8]">
                      {ln.score.toFixed(2)}
                    </span>
                    <span className="text-[9px] text-zinc-400">
                      [{Object.keys(ln.found_by).join(',')}]
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Loading State */}
        {ask.isPending && (
          <div className="mt-6 p-4 rounded-2xl bg-white/[0.03] border border-white/10 flex items-center gap-3 text-xs font-mono text-zinc-300">
            <Loader2 className="w-4 h-4 animate-spin text-[#e5b869] shrink-0" />
            <div>
              <div>
                Traversing 3D knowledge graph & hybrid indices
                {generate ? ' + generating cited answer' : ''}
                {verifyClaims ? ' + running NLI semantic claim verification' : ''}…
              </div>
              <div className="text-[10px] text-zinc-500 mt-0.5">
                Watch the 3D neural cosmos highlight active seed nodes when retrieval completes.
              </div>
            </div>
          </div>
        )}

        {/* Error State */}
        {ask.isError && (
          <div className="mt-5 p-3.5 rounded-2xl bg-[#d9534f]/10 border border-[#d9534f]/30 text-xs text-[#f08a86] flex gap-2.5">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <div>Request failed: {ask.error.message}</div>
          </div>
        )}

        {res?.llm_error && (
          <div className="mt-5 p-3.5 rounded-2xl bg-[#e5b869]/10 border border-[#e5b869]/30 text-xs text-[#f5d799] flex gap-2.5">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <div>
              Retrieved relevant passages & graph facts, but LLM generation returned an error ({res.llm_error}).
            </div>
          </div>
        )}

        {/* Full Query Response */}
        {res && (
          <div className="mt-6 space-y-5">
            {/* Pipeline Telemetry & Routing Ribbon */}
            <div className="flex items-center flex-wrap gap-2 text-[11px] font-mono">
              <span className="px-2.5 py-1 rounded-full bg-[#e5b869]/15 border border-[#e5b869]/40 text-[#e5b869] uppercase font-semibold">
                Path: {res.path_taken}
              </span>
              {res.intent && (
                <span className="px-2.5 py-1 rounded-full bg-[#7ba7bc]/15 border border-[#7ba7bc]/40 text-[#a8cadb] uppercase">
                  Intent: {res.intent}
                </span>
              )}
              {res.llm_provider && (
                <span className="px-2.5 py-1 rounded-full bg-white/8 border border-white/15 text-zinc-200">
                  LLM: {res.llm_provider} ({res.llm_model})
                </span>
              )}
              {res.graph && res.graph.seeds.length > 0 && (
                <span className="px-2.5 py-1 rounded-full bg-[#5ec2a8]/15 border border-[#5ec2a8]/40 text-[#5ec2a8]">
                  {res.graph.seeds.length} Graph Seeds · {res.graph.hops} Hop(s) · +{res.graph.extra_chunks} Graph Chunks
                </span>
              )}
              <span className="px-2.5 py-1 rounded-full bg-black/40 border border-white/10 text-zinc-400">
                {res.timing.total_ms} ms total ({res.timing.total_retrieval_ms ?? 0} ms retrieval)
              </span>
            </div>

            {/* Sub-question Decomposition / Community Overviews Used */}
            {((res.question_parts && res.question_parts.length > 1) ||
              (res.overview_used && res.overview_used.length > 0)) && (
              <div className="p-3 rounded-2xl bg-white/[0.02] border border-white/10 text-xs font-mono space-y-1.5">
                {res.question_parts && res.question_parts.length > 1 && (
                  <div className="text-zinc-300">
                    <span className="text-[#e07a5f] uppercase mr-2">Decomposed Sub-Queries:</span>
                    {res.question_parts.join('  |  ')}
                  </div>
                )}
                {res.overview_used && res.overview_used.length > 0 && (
                  <div className="text-zinc-300">
                    <span className="text-[#7ba7bc] uppercase mr-2">Community Galaxies Matched:</span>
                    {res.overview_used.join(' · ')}
                  </div>
                )}
              </div>
            )}

            {/* Answer Card */}
            {res.answer != null && (
              <section className="glass-panel-glow rounded-2xl p-5">
                <div className="flex items-center justify-between gap-2 mb-3">
                  <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-[#e5b869]">
                    <ShieldCheck className="w-4 h-4" />
                    <span>
                      {res.source === 'general_knowledge'
                        ? 'General Knowledge Answer (Uncovered by Corpus)'
                        : 'Grounded Synthesis'}
                    </span>
                  </div>
                  {res.verification?.available && (
                    <div className="flex items-center gap-2 text-[11px] font-mono">
                      <span className="px-2 py-0.5 rounded bg-[#5ec2a8]/15 text-[#5ec2a8] border border-[#5ec2a8]/30">
                        {res.verification.supported} NLI Supported
                      </span>
                      {(res.verification.unverified ?? 0) > 0 && (
                        <span className="px-2 py-0.5 rounded bg-white/10 text-zinc-300">
                          {res.verification.unverified} Unverified
                        </span>
                      )}
                      {(res.verification.contradicted ?? 0) > 0 && (
                        <span className="px-2 py-0.5 rounded bg-[#d9534f]/20 text-[#f08a86] border border-[#d9534f]/40">
                          {res.verification.contradicted} Contradicted
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {res.refused ? (
                  <p className="text-zinc-300 text-sm">{res.answer}</p>
                ) : (
                  <AnswerText text={res.answer} nSources={res.sources.length} />
                )}

                {res.invalid_citations.length > 0 && (
                  <p className="mt-3 text-xs text-[#e5b869]">
                    Stripped nonexistent source markers: {res.invalid_citations.join(', ')}
                  </p>
                )}

                {res.uncited_sentences.length > 0 && (
                  <details className="mt-3 text-xs text-[#e5b869]">
                    <summary className="cursor-pointer font-mono">
                      {res.uncited_sentences.length} sentence(s) without explicit [n] citation
                    </summary>
                    <ul className="mt-1.5 list-disc pl-5 text-zinc-400 space-y-1">
                      {res.uncited_sentences.map((s, i) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ul>
                  </details>
                )}

                {/* NLI Claim-by-Claim Semantic Verification Matrix */}
                {res.verification?.checks && res.verification.checks.length > 0 && (
                  <details className="mt-4 pt-3 border-t border-white/10 text-xs">
                    <summary className="cursor-pointer font-mono text-[11px] text-[#e5b869] flex items-center justify-between">
                      <span>
                        Inspect NLI Cross-Encoder Claim Verification ({res.verification.checks.length} claims checked)
                      </span>
                      <span className="text-zinc-400">Click to expand</span>
                    </summary>
                    <div className="mt-3 space-y-2">
                      {res.verification.checks.map((chk, idx) => (
                        <div
                          key={idx}
                          className="p-2.5 rounded-xl bg-black/35 border border-white/8 flex items-start justify-between gap-3"
                        >
                          <div className="flex items-start gap-2 min-w-0">
                            {chk.verdict === 'supported' && (
                              <CheckCircle2 className="w-4 h-4 text-[#5ec2a8] shrink-0 mt-0.5" />
                            )}
                            {chk.verdict === 'unverified' && (
                              <HelpCircle className="w-4 h-4 text-zinc-400 shrink-0 mt-0.5" />
                            )}
                            {chk.verdict === 'contradicted' && (
                              <XCircle className="w-4 h-4 text-[#d9534f] shrink-0 mt-0.5" />
                            )}
                            <div className="text-zinc-200 text-xs leading-relaxed">
                              {chk.sentence}
                            </div>
                          </div>
                          <div className="shrink-0 font-mono text-[10px] text-right">
                            <div
                              className={
                                chk.verdict === 'supported'
                                  ? 'text-[#5ec2a8]'
                                  : chk.verdict === 'contradicted'
                                  ? 'text-[#f08a86]'
                                  : 'text-zinc-400'
                              }
                            >
                              {chk.verdict.toUpperCase()}
                            </div>
                            <div className="text-zinc-500">
                              ent {(chk.entailment * 100).toFixed(0)}% · con {(chk.contradiction * 100).toFixed(0)}%
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </section>
            )}

            {/* 3D Graph Seeds & Multi-Hop Evidence Trail */}
            {((res.graph?.seeds && res.graph.seeds.length > 0) ||
              (res.evidence_trail && res.evidence_trail.length > 0)) && (
              <section className="rounded-2xl p-4 glass-panel-subtle border border-[#5ec2a8]/25 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#5ec2a8]">
                    <GitBranch className="w-4 h-4" />
                    <span>3D Graph Context & Evidence Trail</span>
                  </div>
                  <span className="text-[10px] font-mono text-zinc-400">
                    Click any entity to fly in 3D
                  </span>
                </div>

                {res.graph?.seeds && res.graph.seeds.length > 0 && (
                  <div className="flex items-center flex-wrap gap-1.5">
                    <span className="text-[10px] font-mono uppercase text-zinc-400 mr-1">
                      Seeds:
                    </span>
                    {res.graph.seeds.map((seed) => (
                      <button
                        key={seed}
                        onClick={() => onFocusNodeByName(seed)}
                        className="px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-[#5ec2a8]/15 hover:bg-[#5ec2a8]/30 text-[#8ee3cb] border border-[#5ec2a8]/40 transition-all cursor-pointer"
                      >
                        {seed}
                      </button>
                    ))}
                  </div>
                )}

                {res.evidence_trail && res.evidence_trail.length > 0 && (
                  <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                    {res.evidence_trail.map((step, i) => (
                      <div
                        key={i}
                        className="p-2.5 rounded-xl bg-black/35 border border-white/8 text-xs space-y-1"
                      >
                        <div className="flex items-center justify-between font-mono text-[11px]">
                          <div className="flex items-center gap-2">
                            <span className="px-1.5 py-0.2 rounded bg-[#e5b869]/15 text-[#e5b869] text-[10px]">
                              Hop {step.hop}
                            </span>
                            <span className="text-[#f3efe6] font-semibold">{step.fact}</span>
                          </div>
                          <a
                            href={`#source-${step.source}`}
                            className="text-[#e5b869] hover:underline"
                          >
                            [{step.source}]
                          </a>
                        </div>
                        {step.quote && (
                          <div className="text-[11px] text-zinc-400 italic">"{step.quote}"</div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </section>
            )}

            {/* Retrieved Sources */}
            <section>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-zinc-300">
                  <FileText className="w-4 h-4 text-[#e5b869]" />
                  <span>Retrieved Passages ({res.sources.length})</span>
                </div>
                <div className="text-[10px] font-mono text-zinc-400">
                  {res.timing.candidates} candidate pool ·{' '}
                  {res.timing.reranked ? 'cross-encoder reranked' : 'RRF score order'}
                </div>
              </div>
              <div className="space-y-3">
                {res.sources.map((s) => (
                  <SourceCard
                    key={s.chunk_id}
                    s={s}
                    cited={res.cited.map(Number).includes(s.n)}
                  />
                ))}
                {res.sources.length === 0 && (
                  <p className="text-xs text-zinc-500 font-mono">
                    No passages passed the relevance calibration gate for this question.
                  </p>
                )}
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  )
}
