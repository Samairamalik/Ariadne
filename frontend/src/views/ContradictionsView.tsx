import { useState, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import {
  AlertTriangle,
  Route,
  ArrowRight,
  Loader2,
  Sparkles,
  Quote,
  Eye,
  ShieldAlert,
  Zap,
} from 'lucide-react'
import { graphApi } from '../api/client'
import { categorizePredicate } from '../components/Graph3D'
import type { GraphEdge, GraphNode, EntityPathResponse } from '../types'

interface Props {
  nodes: GraphNode[]
  edges: GraphEdge[]
  initialEntityA?: string
  onHighlightGraph: (nodeNames: string[], edgePairs: string[], label: string) => void
  onFocusNodeByName: (name: string) => void
  onSwitchToPure3D: () => void
}

export function ContradictionsView({
  nodes,
  edges,
  initialEntityA = '',
  onHighlightGraph,
  onFocusNodeByName,
  onSwitchToPure3D,
}: Props) {
  const [activeTab, setActiveTab] = useState<'path' | 'tensions'>('path')
  const [entityA, setEntityA] = useState(initialEntityA || 'BERT')
  const [entityB, setEntityB] = useState(
    (initialEntityA || 'BERT').toLowerCase().includes('transformer') ? 'BERT' : 'Transformer'
  )
  const [maxHops, setMaxHops] = useState(3)

  const byId = new Map(nodes.map((n) => [n.id, n]))
  const resolveName = (x: string | GraphNode) =>
    typeof x === 'string' ? byId.get(x)?.name ?? x : x.name

  // Top hub entities for quick 1-click path selection
  const topHubs = [...nodes]
    .sort((a, b) => (b.degree || 0) - (a.degree || 0))
    .slice(0, 12)

  // Both explicit CONTRADICTS and natural-language TENSION / PERFORMANCE comparison edges
  const tensionEdges = edges.filter((e) => {
    const p = (e.predicate || e.type || '').toLowerCase()
    const cat = categorizePredicate(p)
    return (
      cat === 'TENSION' ||
      p.includes('outperform') ||
      p.includes('compar') ||
      p.includes('replac') ||
      p === 'contradicts'
    )
  })

  const tracePath = useMutation<EntityPathResponse, Error, { a?: string; b?: string } | void>({
    mutationFn: (vars) =>
      graphApi.findPath(
        vars && typeof vars === 'object' && vars.a ? vars.a : entityA,
        vars && typeof vars === 'object' && vars.b ? vars.b : entityB,
        maxHops
      ),
    onSuccess: (data) => {
      if (data.connected && data.paths.length > 0) {
        const names = new Set<string>()
        const pairs = new Set<string>()
        data.paths.forEach((p) => {
          p.steps.forEach((s) => {
            names.add(s.from.toLowerCase())
            names.add(s.to.toLowerCase())
            pairs.add(`${s.from.toLowerCase()}::${s.to.toLowerCase()}`)
          })
        })
        onHighlightGraph(
          Array.from(names),
          Array.from(pairs),
          `Synapse Bridge: ${data.from} ↔ ${data.to} (${data.paths[0].hops} hop${data.paths[0].hops > 1 ? 's' : ''})`
        )
      }
    },
  })

  useEffect(() => {
    const startA = initialEntityA || entityA || 'BERT'
    const nextB =
      startA.toLowerCase() === entityB.toLowerCase()
        ? startA.toLowerCase().includes('bert')
          ? 'Transformer'
          : 'BERT'
        : entityB
    if (initialEntityA) {
      setEntityA(startA)
      setEntityB(nextB)
      setActiveTab('path')
    }
    if (startA.trim().length >= 2 && nextB.trim().length >= 2) {
      tracePath.mutate({ a: startA, b: nextB })
    }
  }, [initialEntityA])

  return (
    <div className="fixed top-16 bottom-5 left-1/2 -translate-x-1/2 w-[calc(100vw-2rem)] max-w-4xl z-30 overflow-y-auto px-4 pt-3 pb-8 pointer-events-auto">
      <div className="glass-panel rounded-3xl p-6 border border-[#e5b869]/18 shadow-[0_20px_70px_rgba(0,0,0,0.75)] backdrop-blur-2xl">
        {/* Header & Mode Switch */}
        <div className="flex items-center justify-between flex-wrap gap-3 mb-5">
          <div>
            <h1 className="font-brand text-xl font-bold text-[#f3efe6] flex items-center gap-2">
              <Route className="w-5 h-5 text-[#5ec2a8]" />
              <span>Synapse Path Finder & Paradigm Tensions</span>
            </h1>
            <p className="text-xs text-zinc-400 mt-0.5">
              Discover multi-hop chains of evidence between any two entities or inspect conflicting/comparative claims across papers.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex items-center p-1 rounded-full bg-black/40 border border-white/10 text-xs font-mono">
              <button
                onClick={() => setActiveTab('path')}
                className={`px-3 py-1 rounded-full transition-all cursor-pointer ${
                  activeTab === 'path'
                    ? 'bg-[#5ec2a8]/20 text-[#5ec2a8] border border-[#5ec2a8]/40 font-semibold'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                Synapse Bridge
              </button>
              <button
                onClick={() => setActiveTab('tensions')}
                className={`px-3 py-1 rounded-full transition-all cursor-pointer ${
                  activeTab === 'tensions'
                    ? 'bg-[#d9534f]/20 text-[#f08a86] border border-[#d9534f]/40 font-semibold'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                Tensions & Comparisons ({tensionEdges.length})
              </button>
            </div>

            <button
              onClick={onSwitchToPure3D}
              className="px-3 py-1.5 rounded-full text-[11px] font-mono bg-[#e5b869]/15 hover:bg-[#e5b869]/25 border border-[#e5b869]/40 text-[#e5b869] flex items-center gap-1.5 cursor-pointer"
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Full 3D</span>
            </button>
          </div>
        </div>

        {/* TAB 1: Multi-Hop Synapse Path Finder (/api/graph/path) */}
        {activeTab === 'path' && (
          <div className="space-y-5">
            {/* Quick Hub Entity Chips */}
            {topHubs.length > 0 && (
              <div>
                <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 mb-1.5">
                  Quick-Fill Hub Entities (Click = Origin, Shift+Click = Target):
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {topHubs.map((h) => (
                    <button
                      key={h.id}
                      onClick={(e) => {
                        if (e.shiftKey) setEntityB(h.name)
                        else setEntityA(h.name)
                      }}
                      className={`px-2.5 py-1 rounded-lg text-[11px] font-mono border transition-all cursor-pointer ${
                        entityA.toLowerCase() === h.name.toLowerCase()
                          ? 'bg-[#5ec2a8]/20 text-[#5ec2a8] border-[#5ec2a8]/50'
                          : entityB.toLowerCase() === h.name.toLowerCase()
                          ? 'bg-[#e5b869]/20 text-[#e5b869] border-[#e5b869]/50'
                          : 'bg-white/5 text-zinc-300 border-white/10 hover:bg-white/10'
                      }`}
                    >
                      {h.name}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Path Controls */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-end bg-black/35 p-4 rounded-2xl border border-white/10">
              <div className="md:col-span-4">
                <label className="block text-[10px] font-mono uppercase text-[#5ec2a8] mb-1">
                  Origin Entity (A)
                </label>
                <input
                  type="text"
                  value={entityA}
                  onChange={(e) => setEntityA(e.target.value)}
                  placeholder="e.g. BERT"
                  className="w-full px-3 py-2 rounded-xl bg-black/50 border border-white/15 text-xs font-mono text-[#f3efe6] focus:outline-none focus:border-[#5ec2a8]"
                />
              </div>

              <div className="md:col-span-4">
                <label className="block text-[10px] font-mono uppercase text-[#e5b869] mb-1">
                  Target Entity (B)
                </label>
                <input
                  type="text"
                  value={entityB}
                  onChange={(e) => setEntityB(e.target.value)}
                  placeholder="e.g. GLUE"
                  className="w-full px-3 py-2 rounded-xl bg-black/50 border border-white/15 text-xs font-mono text-[#f3efe6] focus:outline-none focus:border-[#e5b869]"
                />
              </div>

              <div className="md:col-span-2">
                <label className="block text-[10px] font-mono uppercase text-zinc-400 mb-1">
                  Max Hops ({maxHops})
                </label>
                <select
                  value={maxHops}
                  onChange={(e) => setMaxHops(Number(e.target.value))}
                  className="w-full px-2.5 py-2 rounded-xl bg-black/50 border border-white/15 text-xs font-mono text-zinc-200 focus:outline-none"
                >
                  <option value={1}>1 Hop</option>
                  <option value={2}>2 Hops</option>
                  <option value={3}>3 Hops</option>
                  <option value={4}>4 Hops</option>
                </select>
              </div>

              <div className="md:col-span-2">
                <button
                  onClick={() => tracePath.mutate()}
                  disabled={
                    tracePath.isPending ||
                    entityA.trim().length < 2 ||
                    entityB.trim().length < 2
                  }
                  className="w-full py-2 px-3 rounded-xl bg-gradient-to-r from-[#5ec2a8] to-[#e5b869] text-[#080a0a] font-bold text-xs flex items-center justify-center gap-1.5 shadow-[0_0_20px_rgba(94,194,168,0.3)] hover:opacity-95 disabled:opacity-40 cursor-pointer"
                >
                  {tracePath.isPending ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Zap className="w-4 h-4" />
                  )}
                  <span>Trace</span>
                </button>
              </div>
            </div>

            {tracePath.isError && (
              <div className="p-3.5 rounded-2xl bg-[#d9534f]/10 border border-[#d9534f]/30 text-xs text-[#f08a86] flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>
                  {(tracePath.error as any)?.response?.data?.detail ||
                    tracePath.error.message}
                </span>
              </div>
            )}

            {tracePath.data && (
              <div className="space-y-4">
                {!tracePath.data.connected || tracePath.data.paths.length === 0 ? (
                  <div className="p-6 rounded-2xl bg-white/[0.02] border border-white/10 text-center text-xs text-zinc-400">
                    No connecting chain found between <b>{tracePath.data.from}</b> and{' '}
                    <b>{tracePath.data.to}</b> within {tracePath.data.max_hops} hops. Try increasing Max Hops to 4.
                  </div>
                ) : (
                  tracePath.data.paths.map((pathObj, pIdx) => (
                    <div
                      key={pIdx}
                      className="rounded-2xl p-4 glass-panel-subtle border border-[#5ec2a8]/35 space-y-3"
                    >
                      <div className="flex items-center justify-between text-xs font-mono">
                        <span className="text-[#5ec2a8] font-bold flex items-center gap-1.5">
                          <Sparkles className="w-3.5 h-3.5" />
                          Chain #{pIdx + 1} · {pathObj.hops} Hop{pathObj.hops > 1 ? 's' : ''}
                        </span>
                        <span className="text-zinc-400 text-[10px]">
                          Highlighted in 3D Cosmos
                        </span>
                      </div>

                      <div className="space-y-2.5">
                        {pathObj.steps.map((step, sIdx) => (
                          <div
                            key={sIdx}
                            className="p-3 rounded-xl bg-black/40 border border-white/8 text-xs space-y-1.5"
                          >
                            <div className="flex items-center flex-wrap gap-2 font-mono text-[11px]">
                              <span className="px-1.5 py-0.2 rounded bg-white/10 text-zinc-300 text-[10px]">
                                Step {sIdx + 1}
                              </span>
                              <button
                                onClick={() => onFocusNodeByName(step.subject)}
                                className="text-[#f3efe6] hover:text-[#5ec2a8] font-semibold cursor-pointer"
                              >
                                {step.subject}
                              </button>
                              <span className="px-2 py-0.5 rounded bg-[#e5b869]/15 text-[#e5b869] border border-[#e5b869]/30">
                                {step.predicate}
                              </span>
                              <ArrowRight className="w-3 h-3 text-zinc-500" />
                              <button
                                onClick={() => onFocusNodeByName(step.object)}
                                className="text-[#f3efe6] hover:text-[#5ec2a8] font-semibold cursor-pointer"
                              >
                                {step.object}
                              </button>
                            </div>
                            {step.quote && (
                              <p className="text-[11.5px] text-zinc-300 italic flex items-start gap-1.5">
                                <Quote className="w-3.5 h-3.5 text-[#5ec2a8] shrink-0 mt-0.5" />
                                <span>"{step.quote}"</span>
                              </p>
                            )}
                            <div className="text-[10px] font-mono text-zinc-500">
                              {step.document}
                              {step.section ? ` · ${step.section}` : ''}
                              {step.page != null ? ` · p.${step.page}` : ''}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        )}

        {/* TAB 2: Tensions, Contradictions & Paradigm Comparisons */}
        {activeTab === 'tensions' && (
          <div className="space-y-3">
            {tensionEdges.length === 0 ? (
              <div className="rounded-2xl p-8 text-center text-xs text-zinc-400 bg-white/[0.02] border border-white/8">
                <AlertTriangle className="w-6 h-6 mx-auto mb-2 text-zinc-500" />
                No comparative or conflicting edges found in the current graph.
              </div>
            ) : (
              tensionEdges.map((e) => {
                const srcName = resolveName(e.source)
                const tgtName = resolveName(e.target)
                const cat = categorizePredicate(e.predicate)
                const isStrictTension = cat === 'TENSION'
                return (
                  <div
                    key={e.id}
                    onClick={() => {
                      onHighlightGraph(
                        [srcName.toLowerCase(), tgtName.toLowerCase()],
                        [`${srcName.toLowerCase()}::${tgtName.toLowerCase()}`],
                        `Tension: ${srcName} [${e.predicate}] ${tgtName}`
                      )
                    }}
                    className={`rounded-2xl p-4 glass-panel-subtle border transition-all cursor-pointer ${
                      isStrictTension
                        ? 'border-[#d9534f]/35 hover:border-[#d9534f]/70'
                        : 'border-[#5ec2a8]/25 hover:border-[#5ec2a8]/60'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 flex-wrap">
                      <div className="flex items-center gap-2 text-sm text-[#f3efe6]">
                        <ShieldAlert
                          className={`w-4 h-4 ${
                            isStrictTension ? 'text-[#d9534f]' : 'text-[#5ec2a8]'
                          }`}
                        />
                        <button
                          onClick={(ev) => {
                            ev.stopPropagation()
                            onFocusNodeByName(srcName)
                          }}
                          className="font-bold hover:text-[#e5b869] cursor-pointer"
                        >
                          {srcName}
                        </button>
                        <span
                          className={`px-2 py-0.5 rounded font-mono text-[10px] uppercase ${
                            isStrictTension
                              ? 'bg-[#d9534f]/20 text-[#f08a86] border border-[#d9534f]/40'
                              : 'bg-[#5ec2a8]/15 text-[#5ec2a8] border border-[#5ec2a8]/35'
                          }`}
                        >
                          {e.predicate}
                        </span>
                        <button
                          onClick={(ev) => {
                            ev.stopPropagation()
                            onFocusNodeByName(tgtName)
                          }}
                          className="font-bold hover:text-[#e5b869] cursor-pointer"
                        >
                          {tgtName}
                        </button>
                      </div>
                      <span className="text-[10px] font-mono text-zinc-500">
                        Click to spotlight in 3D
                      </span>
                    </div>
                    {e.evidence && (
                      <blockquote className="mt-2.5 pl-3 border-l-2 border-[#e07a5f]/60 text-xs text-zinc-300 italic">
                        "{e.evidence}"
                      </blockquote>
                    )}
                    {e.doc_title && (
                      <div className="mt-1.5 text-[10px] font-mono text-zinc-500">
                        Source: {e.doc_title}
                      </div>
                    )}
                  </div>
                )
              })
            )}
          </div>
        )}
      </div>
    </div>
  )
}
