import React, { useState, useEffect } from 'react'
import { 
  X, 
  ExternalLink, 
  Sparkles, 
  Share2, 
  FileText, 
  Layers, 
  ShieldAlert, 
  ArrowRight,
  Bookmark
} from 'lucide-react'
import { graphApi } from '../api/client'
import { NODE_TYPE_COLORS } from './Graph3D'
import type { GraphNode, NodeDetailResponse } from '../types'

interface ConceptDrawerProps {
  node: GraphNode | null
  onClose: () => void
  onSelectNode: (nodeId: string) => void
  onAskAboutNode: (conceptName: string) => void
}

export function ConceptDrawer({
  node,
  onClose,
  onSelectNode,
  onAskAboutNode,
}: ConceptDrawerProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'relationships' | 'evidence'>('overview')
  const [detail, setDetail] = useState<NodeDetailResponse | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!node) {
      setDetail(null)
      return
    }

    setLoading(true)
    graphApi
      .getNodeDetail(node.id)
      .then((res) => {
        setDetail(res)
      })
      .finally(() => {
        setLoading(false)
      })
  }, [node])

  if (!node) return null

  const nodeColor = NODE_TYPE_COLORS[node.type] || '#ff2d9b'

  return (
    <aside className="fixed top-20 right-6 bottom-6 w-96 z-40 flex flex-col rounded-3xl glass-panel shadow-[-20px_0_50px_rgba(0,0,0,0.6)] border border-white/10 backdrop-blur-3xl overflow-hidden pointer-events-auto transition-all animate-in slide-in-from-right duration-300">
      {/* Drawer Header */}
      <div className="p-6 pb-4 border-b border-white/10">
        <div className="flex items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-2">
            <span
              className="w-2.5 h-2.5 rounded-full shadow-[0_0_8px_currentColor]"
              style={{ backgroundColor: nodeColor, color: nodeColor }}
            />
            <span className="text-[11px] font-mono tracking-wider uppercase px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-zinc-300">
              {node.type}
            </span>
            <span className="text-[11px] font-mono text-zinc-500">
              {node.cluster}
            </span>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-full text-zinc-400 hover:text-white hover:bg-white/10 transition-all cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <h2 className="text-2xl font-bold font-brand tracking-tight text-white mb-2 leading-tight">
          {node.name}
        </h2>

        {/* Confidence Gauge Bar */}
        <div className="p-3 rounded-xl bg-white/[0.03] border border-white/5 mt-3">
          <div className="flex items-center justify-between text-xs font-mono mb-1.5">
            <span className="text-zinc-400">Confidence Score</span>
            <span className="text-[#00ff88] font-bold">
              {Math.round((node.confidence || 0.96) * 100)}%
            </span>
          </div>
          <div className="w-full h-1.5 bg-black/40 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-[#00f0ff] to-[#00ff88] rounded-full shadow-[0_0_10px_#00ff88]"
              style={{ width: `${Math.round((node.confidence || 0.96) * 100)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center px-6 border-b border-white/10 text-xs font-mono">
        <button
          onClick={() => setActiveTab('overview')}
          className={`py-3 px-3 border-b-2 transition-all cursor-pointer ${
            activeTab === 'overview'
              ? 'border-[#00f0ff] text-[#00f0ff] font-semibold'
              : 'border-transparent text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Overview
        </button>
        <button
          onClick={() => setActiveTab('relationships')}
          className={`py-3 px-3 border-b-2 transition-all cursor-pointer ${
            activeTab === 'relationships'
              ? 'border-[#00f0ff] text-[#00f0ff] font-semibold'
              : 'border-transparent text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Relationships ({detail?.relationships?.length || node.degree || 0})
        </button>
        <button
          onClick={() => setActiveTab('evidence')}
          className={`py-3 px-3 border-b-2 transition-all cursor-pointer ${
            activeTab === 'evidence'
              ? 'border-[#00f0ff] text-[#00f0ff] font-semibold'
              : 'border-transparent text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Evidence ({detail?.evidence_chunks?.length || 0})
        </button>
      </div>

      {/* Drawer Body */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {activeTab === 'overview' && (
          <div className="space-y-4">
            <div>
              <h4 className="text-[11px] font-mono uppercase tracking-wider text-zinc-500 mb-1.5">
                Description & Synthesis
              </h4>
              <p className="text-xs text-zinc-300 leading-relaxed bg-white/[0.02] p-3 rounded-xl border border-white/5">
                {node.description || 'Concept extracted from indexed corpus.'}
              </p>
            </div>

            {node.aliases && node.aliases.length > 0 && (
              <div>
                <h4 className="text-[11px] font-mono uppercase tracking-wider text-zinc-500 mb-1.5">
                  Known Aliases
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {node.aliases.map((alias) => (
                    <span
                      key={alias}
                      className="px-2 py-0.5 rounded-md text-[11px] font-mono bg-white/5 text-zinc-300 border border-white/10"
                    >
                      {alias}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-3 rounded-xl bg-white/[0.02] border border-white/5">
                <span className="text-zinc-500 block text-[10px] uppercase">Mentions</span>
                <span className="text-lg font-bold text-white mt-1 block">
                  {node.mention_count || 1}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-white/[0.02] border border-white/5">
                <span className="text-zinc-500 block text-[10px] uppercase">Connections</span>
                <span className="text-lg font-bold text-[#00f0ff] mt-1 block">
                  {node.degree || 1}
                </span>
              </div>
            </div>

            {node.doc_title && (
              <div className="p-3 rounded-xl bg-white/[0.02] border border-white/5 text-xs font-mono">
                <span className="text-zinc-500 text-[10px] uppercase block mb-1">Origin Document</span>
                <div className="flex items-center gap-2 text-zinc-300">
                  <FileText className="w-3.5 h-3.5 text-[#ff2d9b]" />
                  <span className="truncate">{node.doc_title}</span>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'relationships' && (
          <div className="space-y-2.5">
            {detail?.relationships && detail.relationships.length > 0 ? (
              detail.relationships.map((rel) => {
                const isTarget = rel.target_name === node.name
                const targetId = isTarget ? rel.source_id : rel.target_id
                const otherName = isTarget ? rel.source_name : rel.target_name
                const isContradiction = rel.predicate === 'CONTRADICTS'

                return (
                  <div
                    key={rel.id}
                    onClick={() => onSelectNode(targetId)}
                    className="group p-3 rounded-xl bg-white/[0.03] hover:bg-white/[0.07] border border-white/5 hover:border-[#00f0ff]/30 transition-all cursor-pointer"
                  >
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span
                        className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                          isContradiction
                            ? 'bg-[#ff1744]/20 text-[#ff6e84] border border-[#ff1744]/30'
                            : 'bg-[#00f0ff]/10 text-[#00f0ff] border border-[#00f0ff]/20'
                        }`}
                      >
                        {rel.predicate}
                      </span>
                      <ArrowRight className="w-3.5 h-3.5 text-zinc-500 group-hover:text-[#00f0ff] group-hover:translate-x-1 transition-all" />
                    </div>
                    <div className="font-semibold text-xs text-white group-hover:text-[#00f0ff] transition-colors">
                      {otherName}
                    </div>
                    {rel.evidence_text && (
                      <p className="text-[11px] text-zinc-400 mt-1 line-clamp-2 italic">
                        "{rel.evidence_text}"
                      </p>
                    )}
                  </div>
                )
              })
            ) : (
              <p className="text-xs text-zinc-500 italic text-center py-6">
                No active relationship links loaded.
              </p>
            )}
          </div>
        )}

        {activeTab === 'evidence' && (
          <div className="space-y-3">
            {detail?.evidence_chunks && detail.evidence_chunks.length > 0 ? (
              detail.evidence_chunks.map((chk) => (
                <div
                  key={chk.id}
                  className="p-3 rounded-xl bg-white/[0.03] border border-white/5 text-xs space-y-1.5"
                >
                  <div className="flex items-center justify-between text-[10px] font-mono text-zinc-400">
                    <span>Chunk #{chk.chunk_index}</span>
                    {chk.page != null && <span>Page {chk.page}</span>}
                  </div>
                  {chk.section_path && (
                    <div className="text-[11px] font-mono text-[#00f0ff] truncate">
                      {chk.section_path}
                    </div>
                  )}
                  <p className="text-zinc-300 text-xs leading-relaxed bg-black/30 p-2.5 rounded-lg border border-white/5">
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

      {/* Drawer Action Footer */}
      <div className="p-4 border-t border-white/10 bg-black/20">
        <button
          onClick={() => onAskAboutNode(node.name)}
          className="w-full py-2.5 rounded-xl bg-gradient-to-r from-[#ff2d9b] to-[#00f0ff] text-white font-medium text-xs flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(255,45,155,0.4)] hover:opacity-95 transition-all cursor-pointer"
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span>Ask Ariadne About {node.name}</span>
        </button>
      </div>
    </aside>
  )
}
