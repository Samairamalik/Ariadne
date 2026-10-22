import React from 'react'
import { 
  Sparkles, 
  Search, 
  BookOpen, 
  Upload, 
  Layers, 
  AlertTriangle, 
  Activity, 
  Cpu
} from 'lucide-react'

interface TopNavProps {
  activeView: 'graph' | 'vault' | 'query' | 'contradictions'
  setActiveView: (view: 'graph' | 'vault' | 'query' | 'contradictions') => void
  searchQuery: string
  setSearchQuery: (q: string) => void
  backendHealthy: boolean
  rerankerStatus: string
  nodeCount: number
  edgeCount: number
  contradictionCount: number
  onUploadClick: () => void
}

export function TopNav({
  activeView,
  setActiveView,
  searchQuery,
  setSearchQuery,
  backendHealthy,
  rerankerStatus,
  nodeCount,
  edgeCount,
  contradictionCount,
  onUploadClick,
}: TopNavProps) {
  return (
    <header className="fixed top-0 left-0 right-0 z-40 px-6 py-3.5 flex items-center justify-between pointer-events-auto">
      {/* Brand Identity & System Telemetry */}
      <div className="flex items-center gap-6 shrink-0">
        <div
          onClick={() => setActiveView('graph')}
          className="flex items-center gap-3 cursor-pointer group"
        >
          <div className="relative w-9 h-9 rounded-xl bg-gradient-to-tr from-[#ff2d9b] via-[#9d00ff] to-[#00f0ff] p-[1.5px] shadow-[0_0_20px_rgba(0,240,255,0.35)] group-hover:shadow-[0_0_30px_rgba(255,45,155,0.6)] transition-all duration-300">
            <div className="w-full h-full bg-[#090814] rounded-[10px] flex items-center justify-center">
              <Sparkles className="w-4.5 h-4.5 text-[#00f0ff] animate-pulse" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-brand font-extrabold text-lg tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-white via-[#e8e4f5] to-[#00f0ff]">
                ARIADNE
              </span>
              <span className="text-[10px] font-mono tracking-widest px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-[#00f0ff]">
                v0.1
              </span>
            </div>
            <p className="text-[10px] font-mono tracking-widest text-[#aea8b9]/60 uppercase">
              Neural Knowledge Graph & Hybrid RAG
            </p>
          </div>
        </div>

        {/* Telemetry Status Pill */}
        <div className="hidden min-[2300px]:flex items-center gap-3 px-3 py-1.5 rounded-full glass-panel-subtle text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${backendHealthy ? 'bg-[#00ff88] shadow-[0_0_8px_#00ff88]' : 'bg-[#ff1744] shadow-[0_0_8px_#ff1744]'} animate-pulse`} />
            <span className="text-zinc-300 text-[11px]">{backendHealthy ? 'API ONLINE' : 'API DISCONNECTED'}</span>
          </div>
          <span className="w-1 h-3 bg-white/10" />
          <div className="flex items-center gap-1 text-[11px] text-[#00f0ff]">
            <Cpu className="w-3.5 h-3.5" />
            <span>RERANKER: {rerankerStatus.toUpperCase()}</span>
          </div>
          <span className="w-1 h-3 bg-white/10" />
          <div className="flex items-center gap-1 text-[11px] text-zinc-400">
            <Layers className="w-3.5 h-3.5 text-[#ff2d9b]" />
            <span>{nodeCount} NODES / {edgeCount} EDGES</span>
          </div>
        </div>
      </div>

      {/* Center Search / Node Probe Bar */}
      <div className="flex-1 min-w-[220px] max-w-md mx-4">
        <div className="relative flex items-center">
          <Search className="absolute left-3.5 w-4 h-4 text-zinc-400 pointer-events-none" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search concepts, models, architectures..."
            className="w-full pl-10 pr-12 py-2 rounded-full glass-panel-subtle text-xs font-mono text-zinc-200 placeholder:text-zinc-500 focus:outline-none focus:border-[#00f0ff]/50 focus:ring-1 focus:ring-[#00f0ff]/30 transition-all shadow-[0_4px_20px_rgba(0,0,0,0.4)]"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3.5 text-xs text-zinc-400 hover:text-white"
            >
              ×
            </button>
          )}
        </div>
      </div>

      {/* Primary Navigation Views & Actions */}
      <div className="flex items-center gap-2.5 shrink-0">
        {/* Navigation Tabs */}
        <div className="flex items-center p-1 rounded-full glass-panel border border-white/10 shadow-lg">
          <button
            onClick={() => setActiveView('graph')}
            className={`px-3.5 py-1.5 rounded-full text-xs font-medium flex items-center gap-1.5 whitespace-nowrap transition-all ${
              activeView === 'graph'
                ? 'bg-gradient-to-r from-[#ff2d9b]/20 to-[#00f0ff]/20 text-white border border-[#00f0ff]/40 shadow-[0_0_15px_rgba(0,240,255,0.2)]'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
            }`}
          >
            <Activity className="w-3.5 h-3.5 text-[#00f0ff]" />
            <span>Graph</span>
          </button>

          <button
            onClick={() => setActiveView('query')}
            className={`px-3.5 py-1.5 rounded-full text-xs font-medium flex items-center gap-1.5 whitespace-nowrap transition-all ${
              activeView === 'query'
                ? 'bg-gradient-to-r from-[#ff2d9b]/20 to-[#00f0ff]/20 text-white border border-[#00f0ff]/40 shadow-[0_0_15px_rgba(0,240,255,0.2)]'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-[#ff2d9b]" />
            <span>Ask Ariadne</span>
          </button>

          <button
            onClick={() => setActiveView('vault')}
            className={`px-3.5 py-1.5 rounded-full text-xs font-medium flex items-center gap-1.5 whitespace-nowrap transition-all ${
              activeView === 'vault'
                ? 'bg-gradient-to-r from-[#ff2d9b]/20 to-[#00f0ff]/20 text-white border border-[#00f0ff]/40 shadow-[0_0_15px_rgba(0,240,255,0.2)]'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5 text-[#00ff88]" />
            <span>Documents</span>
          </button>

          <button
            onClick={() => setActiveView('contradictions')}
            className={`px-3.5 py-1.5 rounded-full text-xs font-medium flex items-center gap-1.5 whitespace-nowrap transition-all relative ${
              activeView === 'contradictions'
                ? 'bg-[#ff1744]/20 text-[#ff6e84] border border-[#ff1744]/40 shadow-[0_0_15px_rgba(255,23,68,0.25)]'
                : 'text-zinc-400 hover:text-[#ff6e84] hover:bg-white/5'
            }`}
          >
            <AlertTriangle className="w-3.5 h-3.5 text-[#ff1744]" />
            <span>Contradictions</span>
            {contradictionCount > 0 && (
              <span className="ml-0.5 px-1.5 py-0.2 rounded-full text-[9px] font-mono bg-[#ff1744] text-white">
                {contradictionCount}
              </span>
            )}
          </button>
        </div>

        {/* Upload Source Action */}
        <button
          onClick={onUploadClick}
          className="px-4 py-2 rounded-full bg-gradient-to-r from-[#ff2d9b] to-[#00f0ff] hover:opacity-95 active:scale-95 text-white font-medium text-xs flex items-center gap-2 whitespace-nowrap shadow-[0_0_20px_rgba(255,45,155,0.4)] transition-all cursor-pointer"
        >
          <Upload className="w-3.5 h-3.5" />
          <span>Upload Source</span>
        </button>
      </div>
    </header>
  )
}
