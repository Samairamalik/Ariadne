import { useState } from 'react'
import { ChevronDown, ChevronUp, Flame, Route } from 'lucide-react'
import { NODE_TYPE_COLORS } from './Graph3D'
import type { GraphNode } from '../types'

interface NeuralTelemetryHUDProps {
  nodes: GraphNode[]
  hoveredNode: GraphNode | null
  selectedNodeId: string | null
  onSelectNode: (node: GraphNode) => void
  onOpenPathFinder: (name: string) => void
}

export function NeuralTelemetryHUD({
  nodes,
  hoveredNode,
  selectedNodeId,
  onSelectNode,
  onOpenPathFinder,
}: NeuralTelemetryHUDProps) {
  // Start collapsed by default so the 3D view is clean and uncluttered on page load
  const [collapsed, setCollapsed] = useState(true)

  const topHubs = [...nodes]
    .sort(
      (a, b) =>
        (b.degree || 0) * 2 +
        (b.mention_count || 0) -
        ((a.degree || 0) * 2 + (a.mention_count || 0))
    )
    .slice(0, 6)

  return (
    <div className="fixed top-14 left-3 z-30 flex flex-col items-start gap-2 pointer-events-auto">
      <div
        className={`rounded-2xl glass-panel border border-[#e5b869]/18 shadow-[0_8px_28px_rgba(0,0,0,0.55)] overflow-hidden transition-all ${
          collapsed ? 'w-auto' : 'w-56'
        }`}
      >
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full px-3 py-1.5 flex items-center justify-between gap-2 cursor-pointer hover:bg-white/5 transition-colors"
        >
          <div className="flex items-center gap-1.5">
            <Flame className="w-3.5 h-3.5 text-[#e07a5f]" />
            <span className="text-[11px] font-mono uppercase tracking-wider text-zinc-200 font-semibold">
              Top Hubs
            </span>
          </div>
          {collapsed ? (
            <ChevronDown className="w-3.5 h-3.5 text-zinc-400" />
          ) : (
            <ChevronUp className="w-3.5 h-3.5 text-zinc-400" />
          )}
        </button>

        {!collapsed && (
          <div className="px-2 pb-2 pt-0.5 space-y-1 border-t border-white/8">
            {topHubs.map((hub) => {
              const color = NODE_TYPE_COLORS[hub.type] || '#e5b869'
              const isSelected = selectedNodeId === hub.id
              return (
                <div
                  key={hub.id}
                  onClick={() => onSelectNode(hub)}
                  className={`group px-2 py-1.5 rounded-xl flex items-center justify-between gap-2 text-xs transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-[#e5b869]/15 border border-[#e5b869]/50 text-[#f3efe6]'
                      : 'hover:bg-white/8 text-zinc-200 border border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}` }}
                    />
                    <span className="truncate font-medium text-[11px]">{hub.name}</span>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    <span className="text-[10px] font-mono text-zinc-400">
                      {hub.degree || 0}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        onOpenPathFinder(hub.name)
                      }}
                      title="Trace Path from this node"
                      className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-white/15 text-[#5ec2a8] transition-opacity cursor-pointer"
                    >
                      <Route className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {hoveredNode && (
        <div className="rounded-xl glass-panel-glow px-3 py-2 border border-[#e5b869]/35 max-w-[230px]">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[9px] font-mono uppercase text-[#e5b869]">
              {hoveredNode.type}
            </span>
            <span className="text-[9px] font-mono text-zinc-400">
              {hoveredNode.degree} links
            </span>
          </div>
          <div className="font-brand font-bold text-xs text-[#f3efe6] truncate mt-0.5">
            {hoveredNode.name}
          </div>
        </div>
      )}
    </div>
  )
}
