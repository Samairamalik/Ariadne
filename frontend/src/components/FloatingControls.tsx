import React from 'react'
import { RotateCw, Maximize2, Compass, Sliders, ShieldAlert, Sparkles, Filter } from 'lucide-react'
import { RELATIONSHIP_COLORS, NODE_TYPE_COLORS } from './Graph3D'
import type { RelationshipType, NodeType } from '../types'

interface FloatingControlsProps {
  activeRelationshipTypes: string[]
  setActiveRelationshipTypes: (types: string[]) => void
  activeNodeTypes: string[]
  setActiveNodeTypes: (types: string[]) => void
  autoOrbit: boolean
  setAutoOrbit: (val: boolean) => void
  repulsion: number
  setRepulsion: (val: number) => void
  onResetCamera: () => void
  selectedCluster: string
  setSelectedCluster: (c: string) => void
  clusters: string[]
}

const ALL_RELATIONSHIPS: RelationshipType[] = [
  'BUILDS_ON',
  'SUPPORTS',
  'CONTRADICTS',
  'REQUIRED_BY',
  'TANGENTIALLY_RELATED',
]

const ALL_NODE_TYPES: { type: NodeType; label: string }[] = [
  { type: 'concept', label: 'Concepts' },
  { type: 'model', label: 'Models' },
  { type: 'method', label: 'Methods' },
  { type: 'dataset', label: 'Datasets' },
  { type: 'paper', label: 'Papers' },
]

export function FloatingControls({
  activeRelationshipTypes,
  setActiveRelationshipTypes,
  activeNodeTypes,
  setActiveNodeTypes,
  autoOrbit,
  setAutoOrbit,
  repulsion,
  setRepulsion,
  onResetCamera,
  selectedCluster,
  setSelectedCluster,
  clusters,
}: FloatingControlsProps) {
  const toggleRelationship = (rel: string) => {
    if (activeRelationshipTypes.includes(rel)) {
      if (activeRelationshipTypes.length > 1) {
        setActiveRelationshipTypes(activeRelationshipTypes.filter((r) => r !== rel))
      }
    } else {
      setActiveRelationshipTypes([...activeRelationshipTypes, rel])
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

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-30 flex flex-col items-center gap-2.5 max-w-4xl w-[92%] pointer-events-auto">
      {/* Node Type & Cluster Quick Filter Pills */}
      <div className="flex items-center flex-wrap justify-center gap-1.5 px-3 py-1.5 rounded-full glass-panel-subtle shadow-xl">
        <span className="text-[10px] font-mono text-zinc-400 uppercase tracking-wider mr-1.5 flex items-center gap-1">
          <Filter className="w-3 h-3 text-[#00f0ff]" /> Type:
        </span>
        {ALL_NODE_TYPES.map(({ type, label }) => {
          const isActive = activeNodeTypes.includes(type)
          const color = NODE_TYPE_COLORS[type]
          return (
            <button
              key={type}
              onClick={() => toggleNodeType(type)}
              className={`px-2.5 py-1 rounded-full text-[11px] font-mono flex items-center gap-1.5 transition-all cursor-pointer ${
                isActive
                  ? 'bg-white/10 text-white border border-white/20'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5 border border-transparent'
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

        {clusters.length > 0 && (
          <select
            value={selectedCluster}
            onChange={(e) => setSelectedCluster(e.target.value)}
            className="ml-2 px-2 py-0.5 rounded-full bg-black/40 border border-white/10 text-[11px] font-mono text-zinc-300 focus:outline-none focus:border-[#00f0ff]"
          >
            <option value="all">All Clusters</option>
            {clusters.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Main Bottom Control Island */}
      <div className="flex items-center justify-between gap-4 px-5 py-2.5 rounded-2xl glass-panel shadow-[0_12px_40px_rgba(0,0,0,0.6)] border border-white/10 w-full backdrop-blur-3xl">
        {/* Relationship Filter Toggles */}
        <div className="flex items-center gap-1.5 overflow-x-auto py-0.5">
          <button
            onClick={() => setActiveRelationshipTypes(ALL_RELATIONSHIPS)}
            className={`px-3 py-1 rounded-lg text-xs font-mono transition-all cursor-pointer ${
              activeRelationshipTypes.length === ALL_RELATIONSHIPS.length
                ? 'bg-white/15 text-white border border-white/20 font-semibold'
                : 'text-zinc-400 hover:text-white hover:bg-white/5'
            }`}
          >
            ALL
          </button>
          {ALL_RELATIONSHIPS.map((rel) => {
            const isActive = activeRelationshipTypes.includes(rel)
            const isContradiction = rel === 'CONTRADICTS'
            return (
              <button
                key={rel}
                onClick={() => toggleRelationship(rel)}
                className={`px-2.5 py-1 rounded-lg text-[11px] font-mono transition-all flex items-center gap-1.5 cursor-pointer ${
                  isActive
                    ? isContradiction
                      ? 'bg-[#ff1744]/25 text-[#ff6e84] border border-[#ff1744]/40 shadow-[0_0_10px_rgba(255,23,68,0.3)]'
                      : 'bg-white/15 text-white border border-white/20 shadow-[0_0_10px_rgba(0,240,255,0.15)]'
                    : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5 border border-transparent'
                }`}
              >
                {isContradiction && <ShieldAlert className="w-3 h-3 text-[#ff1744]" />}
                <span>{rel.replace('_', ' ')}</span>
              </button>
            )
          })}
        </div>

        {/* 3D Visual & Physics Settings */}
        <div className="flex items-center gap-3 border-l border-white/10 pl-4">
          {/* Orbit Toggle */}
          <button
            onClick={() => setAutoOrbit(!autoOrbit)}
            title="Auto-orbit camera"
            className={`px-3 py-1 rounded-lg text-xs font-mono flex items-center gap-1.5 transition-all cursor-pointer ${
              autoOrbit
                ? 'bg-[#00f0ff]/20 text-[#00f0ff] border border-[#00f0ff]/40 shadow-[0_0_12px_rgba(0,240,255,0.3)]'
                : 'text-zinc-400 hover:text-white hover:bg-white/5'
            }`}
          >
            <RotateCw className={`w-3.5 h-3.5 ${autoOrbit ? 'animate-spin' : ''}`} />
            <span>ORBIT</span>
          </button>

          {/* Repulsion / Force Slider */}
          <div className="hidden sm:flex items-center gap-2 text-zinc-400 text-xs font-mono">
            <Sliders className="w-3.5 h-3.5 text-zinc-400" />
            <input
              type="range"
              min="20"
              max="200"
              value={repulsion}
              onChange={(e) => setRepulsion(Number(e.target.value))}
              title="Graph Physics Repulsion"
              className="w-20 accent-[#00f0ff] cursor-pointer h-1 bg-zinc-700 rounded-lg"
            />
          </div>

          {/* Reset Camera */}
          <button
            onClick={onResetCamera}
            title="Reset Camera View"
            className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-white/10 transition-all cursor-pointer"
          >
            <Compass className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  )
}
