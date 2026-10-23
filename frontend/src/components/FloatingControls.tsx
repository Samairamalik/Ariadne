import { useState } from 'react'
import {
  RotateCw,
  Compass,
  Sliders,
  ShieldAlert,
  Sparkles,
  Filter,
  Orbit,
  Globe,
  Columns3,
  Network,
  Tag,
  X,
  Zap,
} from 'lucide-react'
import { NODE_TYPE_COLORS, PREDICATE_CATEGORY_META, type PredicateCategory } from './Graph3D'
import type { NodeType, SpatialFormation, DocumentItem } from '../types'

interface FloatingControlsProps {
  activeRelationshipCategories: string[]
  setActiveRelationshipCategories: (cats: string[]) => void
  activeNodeTypes: string[]
  setActiveNodeTypes: (types: string[]) => void
  autoOrbit: boolean
  setAutoOrbit: (val: boolean) => void
  livingMotion: boolean
  setLivingMotion: (val: boolean) => void
  formation: SpatialFormation
  setFormation: (f: SpatialFormation) => void
  repulsion: number
  setRepulsion: (val: number) => void
  maxNodes: number
  setMaxNodes: (val: number) => void
  totalNodesCount: number
  labelMode: 'smart' | 'all' | 'minimal'
  setLabelMode: (m: 'smart' | 'all' | 'minimal') => void
  onResetCamera: () => void
  selectedCluster: string
  setSelectedCluster: (c: string) => void
  clusters: string[]
  selectedDocId: string
  setSelectedDocId: (id: string) => void
  docs: DocumentItem[]
  activeSpotlightLabel: string | null
  onClearSpotlight: () => void
}

const ALL_CATEGORIES: PredicateCategory[] = [
  'ARCHITECTURE',
  'PERFORMANCE',
  'METHODOLOGY',
  'TENSION',
  'GENERAL',
]

const ALL_NODE_TYPES: { type: NodeType; label: string }[] = [
  { type: 'concept', label: 'Concepts' },
  { type: 'model', label: 'Models' },
  { type: 'method', label: 'Methods' },
  { type: 'dataset', label: 'Datasets' },
  { type: 'paper', label: 'Papers' },
  { type: 'person', label: 'People' },
  { type: 'other', label: 'Other' },
]

const FORMATIONS: { id: SpatialFormation; label: string; icon: any; hint: string }[] = [
  { id: 'neural', label: 'Web', icon: Network, hint: 'Organic 3D force-directed synaptic energy field' },
  { id: 'galaxy', label: 'Spiral', icon: Orbit, hint: '4-arm 3D logarithmic galaxy with hub core' },
  { id: 'sphere', label: 'Dome', icon: Globe, hint: 'Concentric Fibonacci spheres by entity layer' },
  { id: 'pillars', label: 'Towers', icon: Columns3, hint: '3D helical strata grouped by source document' },
]

export function FloatingControls({
  activeRelationshipCategories,
  setActiveRelationshipCategories,
  activeNodeTypes,
  setActiveNodeTypes,
  autoOrbit,
  setAutoOrbit,
  livingMotion,
  setLivingMotion,
  formation,
  setFormation,
  repulsion,
  setRepulsion,
  maxNodes,
  setMaxNodes,
  totalNodesCount,
  labelMode,
  setLabelMode,
  onResetCamera,
  selectedCluster,
  setSelectedCluster,
  clusters,
  selectedDocId,
  setSelectedDocId,
  docs,
  activeSpotlightLabel,
  onClearSpotlight,
}: FloatingControlsProps) {
  const [filtersOpen, setFiltersOpen] = useState(false)

  const toggleCategory = (cat: string) => {
    if (activeRelationshipCategories.includes(cat)) {
      if (activeRelationshipCategories.length > 1) {
        setActiveRelationshipCategories(activeRelationshipCategories.filter((r) => r !== cat))
      }
    } else {
      setActiveRelationshipCategories([...activeRelationshipCategories, cat])
    }
  }

  const toggleNodeType = (type: string) => {
    if (activeNodeTypes.includes(type)) {
      if (activeNodeTypes.length > 1) {
        setActiveNodeTypes(activeNodeTypes.filter((t) => t !== type))
      }
    } else {
      setActiveNodeTypes([...activeNodeTypes, type])
    }
  }

  const cycleLabelMode = () => {
    if (labelMode === 'smart') setLabelMode('all')
    else if (labelMode === 'all') setLabelMode('minimal')
    else setLabelMode('smart')
  }

  const hasCustomFilter =
    activeNodeTypes.length < ALL_NODE_TYPES.length ||
    activeRelationshipCategories.length < ALL_CATEGORIES.length ||
    selectedDocId !== 'all' ||
    selectedCluster !== 'all'

  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-30 flex flex-col items-center gap-2 max-w-[96vw] pointer-events-auto">
      {/* Active Spotlight / Path / RAG Query Banner */}
      {activeSpotlightLabel && (
        <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-[#5ec2a8]/15 border border-[#5ec2a8]/50 text-[#8ee3cb] text-xs font-mono shadow-[0_0_20px_rgba(94,194,168,0.25)] backdrop-blur-xl max-w-full">
          <Sparkles className="w-3.5 h-3.5 text-[#5ec2a8] shrink-0 animate-pulse" />
          <span className="truncate">{activeSpotlightLabel}</span>
          <button
            onClick={onClearSpotlight}
            className="ml-1 p-0.5 rounded-full hover:bg-white/15 text-white cursor-pointer shrink-0"
            title="Clear 3D Spotlight"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Popover Filters & Physics Panel (opens cleanly on demand instead of cluttering the screen) */}
      {filtersOpen && (
        <div className="w-[540px] max-w-[94vw] p-4 rounded-3xl glass-panel border border-[#e5b869]/20 shadow-[0_20px_60px_rgba(0,0,0,0.85)] backdrop-blur-3xl space-y-3.5 text-xs font-mono">
          <div className="flex items-center justify-between border-b border-white/10 pb-2">
            <span className="text-[#f3efe6] font-semibold uppercase tracking-wider text-[11px] flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 text-[#e5b869]" />
              <span>3D Graph Filters & Physics</span>
            </span>
            <div className="flex items-center gap-2">
              {hasCustomFilter && (
                <button
                  onClick={() => {
                    setActiveNodeTypes(ALL_NODE_TYPES.map((n) => n.type))
                    setActiveRelationshipCategories(ALL_CATEGORIES)
                    setSelectedDocId('all')
                    setSelectedCluster('all')
                  }}
                  className="text-[10px] text-[#e5b869] hover:underline cursor-pointer"
                >
                  Reset All
                </button>
              )}
              <button
                onClick={() => setFiltersOpen(false)}
                className="p-1 rounded-full hover:bg-white/10 text-zinc-400 hover:text-white cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Entity Node Types */}
          <div>
            <div className="text-[10px] uppercase text-zinc-400 mb-1.5">Node Types</div>
            <div className="flex flex-wrap gap-1.5">
              {ALL_NODE_TYPES.map(({ type, label }) => {
                const isActive = activeNodeTypes.includes(type)
                const color = NODE_TYPE_COLORS[type]
                return (
                  <button
                    key={type}
                    onClick={() => toggleNodeType(type)}
                    className={`px-2.5 py-1 rounded-full text-[11px] flex items-center gap-1.5 transition-all cursor-pointer ${
                      isActive
                        ? 'bg-white/12 text-[#f3efe6] border border-white/25'
                        : 'text-zinc-500 hover:text-zinc-300 bg-white/[0.02] border border-white/5'
                    }`}
                  >
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: color, boxShadow: isActive ? `0 0 6px ${color}` : 'none' }}
                    />
                    <span>{label}</span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Relationship Categories */}
          <div>
            <div className="text-[10px] uppercase text-zinc-400 mb-1.5">Relationship Categories</div>
            <div className="flex flex-wrap gap-1.5">
              <button
                onClick={() => setActiveRelationshipCategories(ALL_CATEGORIES)}
                className={`px-2.5 py-1 rounded-lg text-[11px] transition-all cursor-pointer ${
                  activeRelationshipCategories.length === ALL_CATEGORIES.length
                    ? 'bg-white/15 text-[#f3efe6] border border-white/25 font-semibold'
                    : 'text-zinc-400 hover:text-white bg-white/5'
                }`}
              >
                All
              </button>
              {ALL_CATEGORIES.map((cat) => {
                const isActive = activeRelationshipCategories.includes(cat)
                const meta = PREDICATE_CATEGORY_META[cat]
                const isTension = cat === 'TENSION'
                return (
                  <button
                    key={cat}
                    onClick={() => toggleCategory(cat)}
                    className={`px-2.5 py-1 rounded-lg text-[11px] transition-all flex items-center gap-1.5 cursor-pointer ${
                      isActive
                        ? isTension
                          ? 'bg-[#d9534f]/25 text-[#f08a86] border border-[#d9534f]/40'
                          : 'bg-white/12 text-[#f3efe6] border border-white/20'
                        : 'text-zinc-500 hover:text-zinc-300 bg-white/[0.02] border border-white/5'
                    }`}
                  >
                    {isTension ? (
                      <ShieldAlert className="w-3 h-3 text-[#d9534f]" />
                    ) : (
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: meta.hex }}
                      />
                    )}
                    <span>{meta.label}</span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Paper & Cluster Selectors + Sliders */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
            {docs.length > 0 && (
              <div>
                <label className="block text-[10px] uppercase text-zinc-400 mb-1">
                  Source Paper
                </label>
                <select
                  value={selectedDocId}
                  onChange={(e) => setSelectedDocId(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-xl bg-black/50 border border-white/15 text-[11px] text-zinc-200 focus:outline-none focus:border-[#e5b869] truncate cursor-pointer"
                >
                  <option value="all">All Papers ({docs.length})</option>
                  {docs.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.title ?? d.filename}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {clusters.length > 0 && (
              <div>
                <label className="block text-[10px] uppercase text-zinc-400 mb-1">
                  Cluster
                </label>
                <select
                  value={selectedCluster}
                  onChange={(e) => setSelectedCluster(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-xl bg-black/50 border border-white/15 text-[11px] text-zinc-200 focus:outline-none focus:border-[#e5b869] cursor-pointer"
                >
                  <option value="all">All Clusters</option>
                  {clusters.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4 pt-1 border-t border-white/10">
            <div>
              <div className="flex justify-between text-[10px] text-zinc-400 mb-1">
                <span>Max Visible Nodes</span>
                <span className="text-[#f3efe6]">{Math.min(maxNodes, totalNodesCount)} / {totalNodesCount}</span>
              </div>
              <input
                type="range"
                min="80"
                max={Math.max(200, totalNodesCount || 600)}
                step="20"
                value={maxNodes}
                onChange={(e) => setMaxNodes(Number(e.target.value))}
                className="w-full accent-[#e5b869] cursor-pointer h-1.5 bg-zinc-700 rounded-lg"
              />
            </div>

            <div>
              <div className="flex justify-between text-[10px] text-zinc-400 mb-1">
                <span>Force Spacing</span>
                <span className="text-[#f3efe6]">{repulsion}</span>
              </div>
              <input
                type="range"
                min="25"
                max="260"
                value={repulsion}
                onChange={(e) => setRepulsion(Number(e.target.value))}
                className="w-full accent-[#5ec2a8] cursor-pointer h-1.5 bg-zinc-700 rounded-lg"
              />
            </div>
          </div>
        </div>
      )}

      {/* Single Clean Bottom Control Bar */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full glass-panel shadow-[0_12px_40px_rgba(0,0,0,0.65)] border border-[#e5b869]/18 backdrop-blur-3xl overflow-x-auto max-w-full">
        {/* 3D Spatial Formations */}
        <div className="flex items-center gap-1 pr-2 border-r border-white/10">
          {FORMATIONS.map(({ id, label, icon: Icon, hint }) => {
            const active = formation === id
            return (
              <button
                key={id}
                onClick={() => setFormation(id)}
                title={hint}
                className={`px-2.5 py-1 rounded-full text-[11px] font-mono flex items-center gap-1.5 transition-all cursor-pointer whitespace-nowrap ${
                  active
                    ? 'bg-gradient-to-r from-[#e5b869]/30 to-[#5ec2a8]/25 text-[#f3efe6] border border-[#e5b869]/50 shadow-[0_0_12px_rgba(229,184,105,0.2)]'
                    : 'text-zinc-400 hover:text-white hover:bg-white/5 border border-transparent'
                }`}
              >
                <Icon className="w-3 h-3 text-[#e5b869]" />
                <span>{label}</span>
              </button>
            )
          })}
        </div>

        {/* Filters & Physics Popover Button */}
        <button
          onClick={() => setFiltersOpen(!filtersOpen)}
          className={`px-2.5 py-1 rounded-full text-[11px] font-mono flex items-center gap-1.5 transition-all cursor-pointer whitespace-nowrap ${
            filtersOpen || hasCustomFilter
              ? 'bg-[#e5b869]/20 text-[#e5b869] border border-[#e5b869]/40'
              : 'text-zinc-300 hover:text-white hover:bg-white/8 border border-transparent'
          }`}
          title="Filter nodes, relationships, papers, and adjust 3D physics"
        >
          <Sliders className="w-3 h-3 text-[#e07a5f]" />
          <span>Filters</span>
          {hasCustomFilter && (
            <span className="w-1.5 h-1.5 rounded-full bg-[#5ec2a8]" />
          )}
        </button>

        {/* Living Pulse Toggle */}
        <button
          onClick={() => setLivingMotion(!livingMotion)}
          title="Toggle Living Neural Drift & Pulse"
          className={`px-2.5 py-1 rounded-full text-[11px] font-mono flex items-center gap-1 transition-all cursor-pointer whitespace-nowrap ${
            livingMotion
              ? 'bg-[#e07a5f]/20 text-[#f2a38e] border border-[#e07a5f]/40'
              : 'text-zinc-400 hover:text-white hover:bg-white/5'
          }`}
        >
          <Zap className="w-3 h-3" />
          <span className="hidden sm:inline">Pulse</span>
        </button>

        {/* Orbit Toggle */}
        <button
          onClick={() => setAutoOrbit(!autoOrbit)}
          title="Auto-orbit 3D camera"
          className={`px-2.5 py-1 rounded-full text-[11px] font-mono flex items-center gap-1 transition-all cursor-pointer whitespace-nowrap ${
            autoOrbit
              ? 'bg-[#5ec2a8]/20 text-[#5ec2a8] border border-[#5ec2a8]/40'
              : 'text-zinc-400 hover:text-white hover:bg-white/5'
          }`}
        >
          <RotateCw className={`w-3 h-3 ${autoOrbit ? 'animate-spin' : ''}`} />
          <span className="hidden sm:inline">Orbit</span>
        </button>

        {/* Label Mode Cycle */}
        <button
          onClick={cycleLabelMode}
          title={`Node Labels: ${labelMode.toUpperCase()} (click to cycle Smart / All / Minimal)`}
          className="px-2.5 py-1 rounded-full text-[11px] font-mono flex items-center gap-1 text-zinc-300 hover:text-white bg-white/5 hover:bg-white/10 border border-white/10 transition-all cursor-pointer whitespace-nowrap"
        >
          <Tag className="w-3 h-3 text-[#e5b869]" />
          <span className="capitalize">{labelMode}</span>
        </button>

        {/* Reset Camera */}
        <button
          onClick={onResetCamera}
          title="Reset 3D Camera View"
          className="p-1.5 rounded-full text-zinc-400 hover:text-white hover:bg-white/10 transition-all cursor-pointer"
        >
          <Compass className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  )
}
