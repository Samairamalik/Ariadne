import { useState, useEffect } from 'react'
import {
  X,
  Sparkles,
  FileText,
  ArrowRight,
  Route,
  Focus,
  BookOpen,
  Quote,
} from 'lucide-react'
import { graphApi } from '../api/client'
import { NODE_TYPE_COLORS, categorizePredicate } from './Graph3D'
import type {
  GraphNode,
  NodeDetailResponse,
  EntityProfileResponse,
} from '../types'

interface ConceptDrawerProps {
  node: GraphNode | null
  allNodes: GraphNode[]
  egoNodeId: string | null
  onToggleEgo: (nodeId: string | null) => void
  onClose: () => void
  onSelectNode: (nodeId: string) => void
  onAskAboutNode: (conceptName: string) => void
  onStartPathFromNode: (conceptName: string) => void
}

export function ConceptDrawer({
  node,
  allNodes,
  egoNodeId,
  onToggleEgo,
  onClose,
  onSelectNode,
  onAskAboutNode,
  onStartPathFromNode,
}: ConceptDrawerProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'dossier' | 'evidence'>('overview')
  const [detail, setDetail] = useState<NodeDetailResponse | null>(null)
  const [profile, setProfile] = useState<EntityProfileResponse | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!node) {
      setDetail(null)
      setProfile(null)
      return
    }

    setLoading(true)
    Promise.all([
      graphApi.getNodeDetail(node.id),
      graphApi.getEntityProfile(node.name),
    ])
      .then(([det, prof]) => {
        setDetail(det)
        setProfile(prof)
      })
      .finally(() => {
        setLoading(false)
      })
  }, [node])

  if (!node) return null

  const nodeColor = NODE_TYPE_COLORS[node.type] || '#e5b869'
  const isEgoActive = egoNodeId === node.id

  const jumpToEntityByName = (name: string) => {
    const match = allNodes.find(
      (n) =>
        n.name.toLowerCase() === name.toLowerCase() ||
        (n.aliases || []).some((a) => a.toLowerCase() === name.toLowerCase())
    )
    if (match) {
      onSelectNode(match.id)
    }
  }

  const totalFacts = profile?.entity?.facts ?? detail?.relationships?.length ?? node.degree ?? 0
  const totalPapers = profile?.entity?.documents ?? (node.doc_title ? 1 : 0)

  return (
    <aside className="fixed top-14 right-3 bottom-16 w-[340px] max-w-[calc(100vw-1.5rem)] z-40 flex flex-col rounded-2xl glass-panel shadow-[-16px_0_50px_rgba(0,0,0,0.75)] border border-[#e5b869]/20 backdrop-blur-3xl overflow-hidden pointer-events-auto">
      {/* Compact Header */}
      <div className="p-3.5 pb-2.5 border-b border-white/10">
        <div className="flex items-center justify-between gap-2 mb-1.5">
          <div className="flex items-center gap-1.5 min-w-0">
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: nodeColor, boxShadow: `0 0 8px ${nodeColor}` }}
            />
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-white/10 text-zinc-200">
              {node.type}
            </span>
            <span className="text-[10px] font-mono text-zinc-400 truncate">
              {node.mention_count || 1}m · {totalFacts} links · {totalPapers || 1}p
            </span>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            <button
              onClick={() => onToggleEgo(isEgoActive ? null : node.id)}
              title={isEgoActive ? 'Exit 2-Hop Isolation' : 'Isolate 2-Hop Neighborhood'}
              className={`px-2 py-0.5 rounded-full text-[10px] font-mono flex items-center gap-1 transition-all cursor-pointer ${
                isEgoActive
                  ? 'bg-[#5ec2a8]/20 text-[#5ec2a8] border border-[#5ec2a8]/50'
                  : 'bg-white/5 text-zinc-300 hover:text-white border border-white/10'
              }`}
            >
              <Focus className="w-3 h-3" />
              <span>{isEgoActive ? 'Isolated' : 'Isolate'}</span>
            </button>
            <button
              onClick={onClose}
              className="p-1 rounded-full text-zinc-400 hover:text-white hover:bg-white/10 transition-all cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <h2 className="text-base font-bold font-brand text-[#f3efe6] leading-snug break-words">
          {node.name}
        </h2>
      </div>

      {/* 3 Clean Tabs (no horizontal scrollbar) */}
      <div className="grid grid-cols-3 border-b border-white/10 text-[11px] font-mono text-center">
        <button
          onClick={() => setActiveTab('overview')}
          className={`py-2 border-b-2 transition-all cursor-pointer ${
            activeTab === 'overview'
              ? 'border-[#e5b869] text-[#e5b869] font-semibold'
              : 'border-transparent text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Overview
        </button>
        <button
          onClick={() => setActiveTab('dossier')}
          className={`py-2 border-b-2 transition-all cursor-pointer ${
            activeTab === 'dossier'
              ? 'border-[#e5b869] text-[#e5b869] font-semibold'
              : 'border-transparent text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Facts ({totalFacts})
        </button>
        <button
          onClick={() => setActiveTab('evidence')}
          className={`py-2 border-b-2 transition-all cursor-pointer ${
            activeTab === 'evidence'
              ? 'border-[#e5b869] text-[#e5b869] font-semibold'
              : 'border-transparent text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Passages ({detail?.evidence_chunks?.length || 0})
        </button>
      </div>

      {/* Scrollable Body */}
      <div className="flex-1 overflow-y-auto p-3.5 space-y-3.5">
        {loading && (
          <div className="text-xs font-mono text-zinc-400 animate-pulse py-2">
            Loading entity profile…
          </div>
        )}

        {activeTab === 'overview' && (
          <div className="space-y-3.5">
            {profile?.related && profile.related.length > 0 && (
              <div>
                <h4 className="text-[10px] font-mono uppercase tracking-wider text-zinc-400 mb-1.5 flex items-center justify-between">
                  <span>Connected Entities</span>
                  <span className="text-[9px] text-[#e5b869]">Click to focus</span>
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {profile.related.map((rel) => (
                    <button
                      key={rel.name}
                      onClick={() => jumpToEntityByName(rel.name)}
                      className="px-2 py-1 rounded-lg text-[11px] font-mono bg-[#e5b869]/10 hover:bg-[#e5b869]/20 text-zinc-100 border border-[#e5b869]/25 flex items-center gap-1 transition-all cursor-pointer"
                    >
                      <span className="truncate max-w-[180px]">{rel.name}</span>
                      <span className="text-[9px] px-1 rounded bg-black/40 text-[#e5b869]">
                        {rel.facts}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Direct Relationships */}
            {detail?.relationships && detail.relationships.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
                  Direct Links
                </h4>
                {detail.relationships.slice(0, 8).map((rel) => {
                  const isTarget = rel.target_name === node.name
                  const targetId = isTarget ? rel.source_id : rel.target_id
                  const otherName = isTarget ? rel.source_name : rel.target_name
                  const cat = categorizePredicate(rel.predicate)
                  const isTension = cat === 'TENSION'

                  return (
                    <div
                      key={rel.id}
                      onClick={() => onSelectNode(targetId)}
                      className="group p-2.5 rounded-xl bg-white/[0.03] hover:bg-white/[0.07] border border-white/5 hover:border-[#e5b869]/30 transition-all cursor-pointer"
                    >
                      <div className="flex items-center justify-between text-xs mb-0.5">
                        <span
                          className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                            isTension
                              ? 'bg-[#d9534f]/20 text-[#f08a86]'
                              : 'bg-[#5ec2a8]/12 text-[#5ec2a8]'
                          }`}
                        >
                          {isTarget ? `← ${rel.predicate}` : `${rel.predicate} →`}
                        </span>
                        <ArrowRight className="w-3 h-3 text-zinc-500 group-hover:text-[#e5b869]" />
                      </div>
                      <div className="font-semibold text-xs text-[#f3efe6] group-hover:text-[#e5b869] break-words">
                        {otherName}
                      </div>
                      {rel.evidence_text && (
                        <p className="text-[11px] text-zinc-400 mt-1 line-clamp-2 italic">
                          "{rel.evidence_text}"
                        </p>
                      )}
                    </div>
                  )
                })}
              </div>
            )}

            {node.doc_title && (
              <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/8 text-xs font-mono">
                <span className="text-zinc-500 text-[9px] uppercase block mb-0.5">Source Paper</span>
                <div className="flex items-center gap-1.5 text-zinc-200">
                  <FileText className="w-3.5 h-3.5 text-[#e07a5f] shrink-0" />
                  <span className="truncate">{node.doc_title}</span>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'dossier' && (
          <div className="space-y-3">
            {profile?.documents && profile.documents.length > 0 ? (
              profile.documents.map((doc) => (
                <div
                  key={doc.document_id}
                  className="rounded-xl bg-white/[0.03] border border-white/10 p-3 space-y-2"
                >
                  <div className="flex items-center gap-1.5 text-xs font-medium text-[#e5b869]">
                    <BookOpen className="w-3.5 h-3.5 shrink-0" />
                    <span className="truncate">{doc.document}</span>
                  </div>
                  <div className="space-y-1.5">
                    {doc.facts.map((f, idx) => (
                      <div
                        key={idx}
                        className="p-2 rounded-lg bg-black/30 border border-white/5 text-xs space-y-1"
                      >
                        <div className="flex items-center flex-wrap gap-1 font-mono text-[10.5px]">
                          <span className="text-[#f3efe6] font-semibold">{f.subject}</span>
                          <span className="px-1.5 py-0.2 rounded bg-[#e07a5f]/15 text-[#f2a38e] text-[10px]">
                            {f.predicate}
                          </span>
                          <span className="text-[#f3efe6] font-semibold">{f.object}</span>
                        </div>
                        {f.quote && (
                          <p className="text-[11px] text-zinc-400 italic leading-relaxed flex gap-1">
                            <Quote className="w-3 h-3 text-zinc-600 shrink-0 mt-0.5" />
                            <span>"{f.quote}"</span>
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))
            ) : (
              <p className="text-xs text-zinc-500 italic text-center py-6">
                No structured facts available for this node.
              </p>
            )}
          </div>
        )}

        {activeTab === 'evidence' && (
          <div className="space-y-2.5">
            {detail?.evidence_chunks && detail.evidence_chunks.length > 0 ? (
              detail.evidence_chunks.map((chk) => (
                <div
                  key={chk.id}
                  className="p-2.5 rounded-xl bg-white/[0.03] border border-white/8 text-xs space-y-1"
                >
                  <div className="flex items-center justify-between text-[10px] font-mono text-zinc-400">
                    <span>Chunk #{chk.chunk_index}</span>
                    {chk.page != null && <span>p.{chk.page}</span>}
                  </div>
                  {chk.doc_title && (
                    <div className="text-[11px] font-medium text-zinc-200 truncate">
                      {chk.doc_title}
                    </div>
                  )}
                  <p className="text-zinc-300 text-[11.5px] leading-relaxed bg-black/30 p-2 rounded-lg border border-white/5">
                    {chk.text}
                  </p>
                </div>
              ))
            ) : (
              <p className="text-xs text-zinc-500 italic text-center py-6">
                No direct indexed passage citations for this concept.
              </p>
            )}
          </div>
        )}
      </div>

      {/* Action Footer */}
      <div className="p-3 border-t border-white/10 bg-black/30 flex items-center gap-2">
        <button
          onClick={() => onStartPathFromNode(node.name)}
          className="px-3 py-2 rounded-xl bg-white/8 hover:bg-white/15 border border-white/15 text-zinc-200 font-mono text-xs flex items-center justify-center gap-1.5 transition-all cursor-pointer shrink-0"
        >
          <Route className="w-3.5 h-3.5 text-[#5ec2a8]" />
          <span>Trace Path</span>
        </button>
        <button
          onClick={() => onAskAboutNode(node.name)}
          className="flex-1 py-2 px-3 rounded-xl bg-gradient-to-r from-[#e5b869] to-[#c8963e] text-[#080a0a] font-semibold text-xs flex items-center justify-center gap-1.5 hover:opacity-95 transition-all cursor-pointer min-w-0 shadow-[0_0_16px_rgba(229,184,105,0.25)]"
        >
          <Sparkles className="w-3.5 h-3.5 shrink-0" />
          <span className="truncate">Ask About {node.name}</span>
        </button>
      </div>
    </aside>
  )
}
