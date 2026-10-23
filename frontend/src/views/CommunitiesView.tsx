import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Orbit, Sparkles, RefreshCw, Loader2, Eye, Layers } from 'lucide-react'
import { graphApi } from '../api/client'
import type { CommunityItem, GraphNode } from '../types'

interface CommunitiesViewProps {
  nodes: GraphNode[]
  onHighlightGraph: (nodeNames: string[], edgePairs: string[], label: string) => void
  onAskAboutTheme: (question: string) => void
  onSwitchToPure3D: () => void
}

export function CommunitiesView({
  nodes,
  onHighlightGraph,
  onAskAboutTheme,
  onSwitchToPure3D,
}: CommunitiesViewProps) {
  const qc = useQueryClient()

  const commsQ = useQuery({
    queryKey: ['communities'],
    queryFn: graphApi.getCommunities,
  })

  const buildMut = useMutation({
    mutationFn: graphApi.buildCommunities,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['communities'] })
    },
  })

  const communities: CommunityItem[] = commsQ.data ?? []

  const spotlightCommunityIn3D = (comm: CommunityItem) => {
    const text = `${comm.title} ${comm.summary}`.toLowerCase()
    const matched = nodes
      .filter((n) => n.name.length >= 3 && text.includes(n.name.toLowerCase()))
      .map((n) => n.name.toLowerCase())

    if (matched.length > 0) {
      onHighlightGraph(matched, [], `Galaxy Theme: ${comm.title} (${matched.length} matched nodes)`)
    }
  }

  return (
    <div className="fixed top-16 bottom-5 left-1/2 -translate-x-1/2 w-[calc(100vw-2rem)] max-w-5xl z-30 overflow-y-auto px-4 pt-3 pb-8 pointer-events-auto">
      <div className="glass-panel rounded-3xl p-6 border border-[#e5b869]/18 shadow-[0_20px_70px_rgba(0,0,0,0.75)] backdrop-blur-2xl">
        {/* Header */}
        <div className="flex items-start justify-between flex-wrap gap-4 mb-6">
          <div>
            <h1 className="font-brand text-xl font-bold text-[#f3efe6] flex items-center gap-2">
              <Orbit className="w-5 h-5 text-[#7ba7bc]" />
              <span>Louvain Knowledge Galaxies ({communities.length})</span>
            </h1>
            <p className="text-xs text-zinc-400 mt-1">
              Densely connected node communities automatically detected via Louvain modularity and synthesized for global corpus questions.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => buildMut.mutate()}
              disabled={buildMut.isPending}
              className="px-3.5 py-1.5 rounded-full text-xs font-mono bg-[#7ba7bc]/20 hover:bg-[#7ba7bc]/30 border border-[#7ba7bc]/50 text-[#d4e8f2] flex items-center gap-1.5 disabled:opacity-50 transition-all cursor-pointer"
            >
              {buildMut.isPending ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <RefreshCw className="w-3.5 h-3.5" />
              )}
              <span>{buildMut.isPending ? 'Synthesizing…' : 'Rebuild Galaxies'}</span>
            </button>

            <button
              onClick={onSwitchToPure3D}
              className="px-3 py-1.5 rounded-full text-[11px] font-mono bg-[#e5b869]/15 hover:bg-[#e5b869]/25 border border-[#e5b869]/40 text-[#e5b869] flex items-center gap-1.5 cursor-pointer"
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Full 3D</span>
            </button>
          </div>
        </div>

        {commsQ.isLoading && (
          <div className="py-12 flex items-center justify-center gap-2 text-xs font-mono text-zinc-400">
            <Loader2 className="w-4 h-4 animate-spin text-[#7ba7bc]" /> Loading thematic galaxies…
          </div>
        )}

        {communities.length === 0 && !commsQ.isLoading && (
          <div className="p-10 rounded-2xl bg-white/[0.02] border border-white/8 text-center space-y-3">
            <Orbit className="w-8 h-8 mx-auto text-[#7ba7bc]" />
            <div className="text-sm font-medium text-[#f3efe6]">No community summaries built yet</div>
            <p className="text-xs text-zinc-400 max-w-md mx-auto">
              Click "Rebuild Galaxies" above to run Louvain clustering on your knowledge graph and generate thematic summaries.
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {communities.map((comm, idx) => (
            <div
              key={idx}
              onClick={() => spotlightCommunityIn3D(comm)}
              className="group rounded-2xl p-4 glass-panel-subtle border border-white/10 hover:border-[#7ba7bc]/50 transition-all flex flex-col justify-between gap-3 cursor-pointer"
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-2">
                  <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-full bg-[#7ba7bc]/15 text-[#a8cadb] border border-[#7ba7bc]/35 flex items-center gap-1">
                    <Layers className="w-3 h-3" />
                    Galaxy #{idx + 1}
                  </span>
                  <span className="text-[11px] font-mono text-[#e5b869]">
                    {comm.size} Nodes
                  </span>
                </div>
                <h3 className="font-brand font-bold text-base text-[#f3efe6] group-hover:text-[#e5b869] transition-colors">
                  {comm.title}
                </h3>
                <p className="text-xs text-zinc-300 leading-relaxed mt-1.5">
                  {comm.summary}
                </p>
              </div>

              <div className="pt-2 border-t border-white/8 flex items-center justify-between text-[11px] font-mono">
                <span className="text-zinc-500 group-hover:text-zinc-300">
                  Click card to spotlight in 3D
                </span>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onAskAboutTheme(
                      `Summarize the key findings and relationships regarding ${comm.title}.`
                    )
                  }}
                  className="text-[#e07a5f] hover:text-[#f3efe6] flex items-center gap-1 cursor-pointer"
                >
                  <Sparkles className="w-3 h-3" />
                  <span>Ask Ariadne</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
