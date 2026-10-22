import { AlertTriangle } from 'lucide-react'
import type { GraphEdge, GraphNode } from '../types'

export function ContradictionsView({ nodes, edges }: { nodes: GraphNode[]; edges: GraphEdge[] }) {
  const byId = new Map(nodes.map((n) => [n.id, n]))
  const name = (x: string | GraphNode) => (typeof x === 'string' ? byId.get(x)?.name ?? x : x.name)
  const items = edges.filter((e) => e.predicate === 'CONTRADICTS')

  return (
    <div className="max-w-3xl mx-auto px-6 pt-24 pb-16 h-screen overflow-y-auto">
      <h1 className="font-headline text-2xl font-semibold text-white mb-1">Contradictions</h1>
      <p className="text-sm text-zinc-400 mb-6">Places where one source challenges a claim made by another.</p>
      {items.length === 0 ? (
        <div className="glass-panel-subtle rounded-2xl p-10 text-center text-sm text-zinc-400">
          <AlertTriangle className="w-6 h-6 mx-auto mb-3 text-zinc-500" />
          Nothing to show. Contradictions are found by relationship extraction, which isn't built yet — this view will
          fill in once documents are processed into a knowledge graph.
        </div>
      ) : (
        <div className="space-y-3">
          {items.map((e) => (
            <div key={e.id} className="glass-panel-subtle rounded-xl p-4 border-[#ff1744]/30">
              <div className="text-sm text-zinc-100">
                <b>{name(e.source)}</b> <span className="text-[#ff6e84] font-mono text-xs">CONTRADICTS</span> <b>{name(e.target)}</b>
              </div>
              <blockquote className="mt-2 pl-3 border-l-2 border-[#ff1744]/50 text-[13px] text-zinc-300">{e.evidence}</blockquote>
              {e.doc_title && <div className="mt-1 text-[11px] font-mono text-zinc-500">{e.doc_title}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
