import { useState, useEffect, useRef } from 'react'
import {
  Sparkles,
  Search,
  BookOpen,
  Upload,
  Activity,
  Route,
  Orbit,
  Radio,
} from 'lucide-react'
import { graphApi } from '../api/client'
import { NODE_TYPE_COLORS } from './Graph3D'
import type { EntitySearchItem, GraphNode } from '../types'

export type AppView = 'graph' | 'query' | 'contradictions' | 'communities' | 'vault'

interface TopNavProps {
  activeView: AppView
  setActiveView: (view: AppView) => void
  searchQuery: string
  setSearchQuery: (q: string) => void
  onSelectSearchResult: (nodeId: string, name: string) => void
  nodes: GraphNode[]
  backendHealthy: boolean
  rerankerStatus: string
  llmLabel: string
  onOpenLLMModal: () => void
  nodeCount: number
  edgeCount: number
  communityCount: number
  contradictionCount: number
  onUploadClick: () => void
}

export function TopNav({
  activeView,
  setActiveView,
  searchQuery,
  setSearchQuery,
  onSelectSearchResult,
  nodes,
  backendHealthy,
  llmLabel,
  onOpenLLMModal,
  nodeCount,
  edgeCount,
  communityCount,
  contradictionCount,
  onUploadClick,
}: TopNavProps) {
  const [results, setResults] = useState<EntitySearchItem[]>([])
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const searchWrapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const q = searchQuery.trim()
    if (q.length < 2) {
      setResults([])
      return
    }
    let cancelled = false
    const timer = setTimeout(async () => {
      try {
        const apiHits = await graphApi.searchEntities(q, 8)
        if (!cancelled) {
          if (apiHits.length > 0) {
            setResults(apiHits)
          } else {
            const low = q.toLowerCase()
            const local = nodes
              .filter((n) => n.name.toLowerCase().includes(low))
              .slice(0, 8)
              .map((n) => ({
                id: n.id,
                name: n.name,
                type: n.type,
                papers: 1,
                mentions: n.mention_count,
                match: 90,
              }))
            setResults(local)
          }
        }
      } catch {
        // fallback local
      }
    }, 180)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [searchQuery, nodes])

  useEffect(() => {
    const onClickOutside = (e: MouseEvent) => {
      if (searchWrapRef.current && !searchWrapRef.current.contains(e.target as Node)) {
        setDropdownOpen(false)
      }
    }
    window.addEventListener('mousedown', onClickOutside)
    return () => window.removeEventListener('mousedown', onClickOutside)
  }, [])

  return (
    <header className="fixed top-0 left-0 right-0 z-40 px-3 py-2 flex items-center justify-between gap-2 w-full box-border pointer-events-auto">
      {/* Left: Brand & Status */}
      <div
        onClick={() => setActiveView('graph')}
        className="flex items-center gap-2 cursor-pointer shrink-0"
      >
        <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-[#e5b869] via-[#c8963e] to-[#5ec2a8] p-[1.5px]">
          <div className="w-full h-full bg-[#080a0a] rounded-[7px] flex items-center justify-center">
            <Sparkles className="w-3.5 h-3.5 text-[#e5b869]" />
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="font-brand font-bold text-sm tracking-wider text-[#f3efe6]">
            ARIADNE
          </span>
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              backendHealthy
                ? 'bg-[#5ec2a8] shadow-[0_0_6px_#5ec2a8]'
                : 'bg-[#d9534f] shadow-[0_0_6px_#d9534f]'
            }`}
            title={backendHealthy ? 'Online' : 'Offline'}
          />
          <span className="hidden lg:inline text-[10px] font-mono text-zinc-400 ml-1">
            {nodeCount}n · {edgeCount}e
          </span>
        </div>
      </div>

      {/* Center: Compact Navigation Pills */}
      <nav className="flex items-center p-0.5 rounded-full glass-panel border border-[#e5b869]/15 shadow-md overflow-x-auto max-w-full">
        <button
          onClick={() => setActiveView('graph')}
          className={`px-2.5 py-1 rounded-full text-[11px] font-medium flex items-center gap-1 whitespace-nowrap transition-all cursor-pointer ${
            activeView === 'graph'
              ? 'bg-gradient-to-r from-[#e5b869]/25 to-[#5ec2a8]/20 text-[#f3efe6] border border-[#e5b869]/50'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
          }`}
        >
          <Activity className="w-3 h-3 text-[#e5b869]" />
          <span>Graph</span>
        </button>

        <button
          onClick={() => setActiveView('query')}
          className={`px-2.5 py-1 rounded-full text-[11px] font-medium flex items-center gap-1 whitespace-nowrap transition-all cursor-pointer ${
            activeView === 'query'
              ? 'bg-gradient-to-r from-[#e5b869]/25 to-[#e07a5f]/20 text-[#f3efe6] border border-[#e5b869]/50'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
          }`}
        >
          <Sparkles className="w-3 h-3 text-[#e07a5f]" />
          <span>Ask AI</span>
        </button>

        <button
          onClick={() => setActiveView('contradictions')}
          className={`px-2.5 py-1 rounded-full text-[11px] font-medium flex items-center gap-1 whitespace-nowrap transition-all cursor-pointer ${
            activeView === 'contradictions'
              ? 'bg-gradient-to-r from-[#5ec2a8]/25 to-[#e5b869]/20 text-[#f3efe6] border border-[#5ec2a8]/50'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
          }`}
        >
          <Route className="w-3 h-3 text-[#5ec2a8]" />
          <span>Paths</span>
          {contradictionCount > 0 && (
            <span className="px-1 rounded-full text-[9px] font-mono bg-[#d9534f]/85 text-white">
              {contradictionCount}
            </span>
          )}
        </button>

        <button
          onClick={() => setActiveView('communities')}
          className={`px-2.5 py-1 rounded-full text-[11px] font-medium flex items-center gap-1 whitespace-nowrap transition-all cursor-pointer ${
            activeView === 'communities'
              ? 'bg-gradient-to-r from-[#7ba7bc]/25 to-[#e5b869]/20 text-[#f3efe6] border border-[#7ba7bc]/50'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
          }`}
        >
          <Orbit className="w-3 h-3 text-[#7ba7bc]" />
          <span>Clusters ({communityCount})</span>
        </button>

        <button
          onClick={() => setActiveView('vault')}
          className={`px-2.5 py-1 rounded-full text-[11px] font-medium flex items-center gap-1 whitespace-nowrap transition-all cursor-pointer ${
            activeView === 'vault'
              ? 'bg-gradient-to-r from-[#e5b869]/25 to-[#95c687]/20 text-[#f3efe6] border border-[#e5b869]/50'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
          }`}
        >
          <BookOpen className="w-3 h-3 text-[#e5b869]" />
          <span>Vault</span>
        </button>
      </nav>

      {/* Right: Compact Search + LLM Switch + Upload */}
      <div className="flex items-center gap-1.5 shrink-0">
        <div ref={searchWrapRef} className="relative w-28 sm:w-40 md:w-48">
          <div className="relative flex items-center">
            <Search className="absolute left-2.5 w-3 h-3 text-zinc-400 pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onFocus={() => setDropdownOpen(true)}
              onChange={(e) => {
                setSearchQuery(e.target.value)
                setDropdownOpen(true)
              }}
              placeholder="Search node..."
              className="w-full pl-7 pr-5 py-1 rounded-full glass-panel-subtle text-[11px] font-mono text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:border-[#e5b869]/60 transition-all"
            />
            {searchQuery && (
              <button
                onClick={() => {
                  setSearchQuery('')
                  setResults([])
                }}
                className="absolute right-2 text-xs text-zinc-400 hover:text-white cursor-pointer"
              >
                ×
              </button>
            )}
          </div>

          {dropdownOpen && results.length > 0 && (
            <div className="absolute top-full right-0 w-64 mt-1.5 rounded-2xl glass-panel border border-[#e5b869]/20 shadow-[0_16px_48px_rgba(0,0,0,0.85)] overflow-hidden z-50 py-1">
              {results.map((item) => {
                const color = NODE_TYPE_COLORS[item.type] || '#e5b869'
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      onSelectSearchResult(item.id, item.name)
                      setDropdownOpen(false)
                    }}
                    className="w-full px-3 py-1.5 text-left hover:bg-white/10 flex items-center justify-between gap-2 transition-colors cursor-pointer"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ backgroundColor: color }}
                      />
                      <span className="text-xs font-medium text-[#f3efe6] truncate">
                        {item.name}
                      </span>
                    </div>
                    <span className="text-[10px] font-mono text-zinc-400 shrink-0">
                      {item.mentions}m
                    </span>
                  </button>
                )
              })}
            </div>
          )}
        </div>

        <button
          onClick={onOpenLLMModal}
          className="px-2.5 py-1 rounded-full glass-panel-subtle border border-white/15 hover:border-[#e5b869]/50 text-[11px] font-mono text-zinc-200 hover:text-white flex items-center gap-1 transition-all cursor-pointer"
          title="Configure Local Ollama or Cloud LLM Router"
        >
          <Radio className="w-3 h-3 text-[#5ec2a8]" />
          <span className="hidden sm:inline">{llmLabel}</span>
        </button>

        <button
          onClick={onUploadClick}
          className="px-2.5 py-1 rounded-full bg-gradient-to-r from-[#e5b869] to-[#c8963e] hover:opacity-95 text-[#080a0a] font-semibold text-[11px] flex items-center gap-1 whitespace-nowrap cursor-pointer shadow-[0_0_16px_rgba(229,184,105,0.28)]"
          title="Upload Document"
        >
          <Upload className="w-3 h-3" />
          <span className="hidden md:inline">Upload</span>
        </button>
      </div>
    </header>
  )
}
