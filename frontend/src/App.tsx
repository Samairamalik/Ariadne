import { useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Loader2, Network, AlertTriangle } from 'lucide-react'
import { documentsApi, graphApi, llmApi, queryApi } from './api/client'
import { TopNav, type AppView } from './components/TopNav'
import { Graph3D, categorizePredicate, type PredicateCategory } from './components/Graph3D'
import { FloatingControls } from './components/FloatingControls'
import { ConceptDrawer } from './components/ConceptDrawer'
import { NeuralTelemetryHUD } from './components/NeuralTelemetryHUD'
import { LLMConfigModal } from './components/LLMConfigModal'
import { QueryView } from './views/QueryView'
import { VaultView } from './views/VaultView'
import { ContradictionsView } from './views/ContradictionsView'
import { CommunitiesView } from './views/CommunitiesView'
import type { GraphNode, SpatialFormation, IngestionJobProgress } from './types'

const ACCEPT = '.pdf,.pptx,.docx,.md,.markdown,.txt,.html,.htm'
const ALL_NODE_TYPES = ['concept', 'model', 'method', 'dataset', 'paper', 'person', 'other']
const ALL_REL_CATEGORIES: PredicateCategory[] = [
  'ARCHITECTURE',
  'PERFORMANCE',
  'METHODOLOGY',
  'TENSION',
  'GENERAL',
]

export default function App() {
  const qc = useQueryClient()
  const fileInput = useRef<HTMLInputElement>(null)

  const [view, setView] = useState<AppView>('graph')
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [focusedId, setFocusedId] = useState<string | null>(null)
  const [egoNodeId, setEgoNodeId] = useState<string | null>(null)

  // Clean, spacious defaults: autoOrbit off so user can freely zoom/rotate right away
  const [relCategories, setRelCategories] = useState<string[]>(ALL_REL_CATEGORIES)
  const [nodeTypes, setNodeTypes] = useState<string[]>(ALL_NODE_TYPES)
  const [cluster, setCluster] = useState('all')
  const [selectedDocId, setSelectedDocId] = useState('all')
  const [autoOrbit, setAutoOrbit] = useState(false)
  const [livingMotion, setLivingMotion] = useState(true)
  const [formation, setFormation] = useState<SpatialFormation>('neural')
  const [repulsion, setRepulsion] = useState(140)
  const [maxNodes, setMaxNodes] = useState(180)
  const [labelMode, setLabelMode] = useState<'smart' | 'all' | 'minimal'>('smart')
  const [cameraKey, setCameraKey] = useState(0)

  const [highlightedNodeNames, setHighlightedNodeNames] = useState<Set<string>>(new Set())
  const [highlightedEdgePairs, setHighlightedEdgePairs] = useState<Set<string>>(new Set())
  const [spotlightLabel, setSpotlightLabel] = useState<string | null>(null)
  const [askSeed, setAskSeed] = useState('')
  const [askKey, setAskKey] = useState(0)
  const [pathSeedA, setPathSeedA] = useState('')
  const [pathKey, setPathKey] = useState(0)
  const [llmModalOpen, setLlmModalOpen] = useState(false)
  const [jobStreams, setJobStreams] = useState<Record<string, IngestionJobProgress>>({})
  const [toast, setToast] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null)

  const health = useQuery({
    queryKey: ['health'],
    queryFn: queryApi.checkHealth,
    refetchInterval: 5000,
  })
  const docsQ = useQuery({
    queryKey: ['documents'],
    queryFn: documentsApi.listDocuments,
    refetchInterval: (q) =>
      q.state.data?.some((d) => !['done', 'failed'].includes(d.status)) ? 1500 : 10000,
  })
  const graphQ = useQuery({
    queryKey: ['graph'],
    queryFn: graphApi.getGraph,
    retry: false,
  })
  const statsQ = useQuery({
    queryKey: ['stats'],
    queryFn: graphApi.getStats,
    retry: false,
    refetchInterval: 15000,
  })
  const commsQ = useQuery({
    queryKey: ['communities'],
    queryFn: graphApi.getCommunities,
    retry: false,
  })
  const llmQ = useQuery({
    queryKey: ['llm-status'],
    queryFn: llmApi.getStatus,
    retry: false,
  })

  const docs = docsQ.data ?? []
  const nodes = graphQ.data?.nodes ?? []
  const edges = graphQ.data?.edges ?? []
  const communities = commsQ.data ?? []

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedId) ?? null,
    [nodes, selectedId]
  )
  const clusters = useMemo(
    () => Array.from(new Set(nodes.map((n) => n.cluster))).sort(),
    [nodes]
  )

  const tensionCount = useMemo(
    () =>
      edges.filter((e) => {
        const p = (e.predicate || e.type || '').toLowerCase()
        return (
          categorizePredicate(p) === 'TENSION' ||
          p.includes('outperform') ||
          p.includes('compar') ||
          p.includes('replac')
        )
      }).length,
    [edges]
  )

  const upload = useMutation({
    mutationFn: (file: File) => documentsApi.uploadDocument(file),
    onSuccess: (r, file) => {
      setToast({
        kind: 'ok',
        text: r.duplicate
          ? `"${file.name}" is already indexed.`
          : `Uploading "${file.name}" — ingestion started.`,
      })
      qc.invalidateQueries({ queryKey: ['documents'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
      setView('vault')

      if (r.job_id && !r.duplicate) {
        documentsApi.streamJobStatus(
          r.job_id,
          (prog) => {
            setJobStreams((prev) => ({ ...prev, [r.job_id]: prog }))
          },
          () => {
            setJobStreams((prev) => {
              const next = { ...prev }
              delete next[r.job_id]
              return next
            })
            qc.invalidateQueries({ queryKey: ['documents'] })
            qc.invalidateQueries({ queryKey: ['graph'] })
            qc.invalidateQueries({ queryKey: ['stats'] })
            qc.invalidateQueries({ queryKey: ['communities'] })
          }
        )
      }
    },
    onError: (e: unknown) => {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data
        ?.detail
      setToast({
        kind: 'err',
        text: detail ?? (e as Error).message ?? 'Upload failed',
      })
    },
  })

  const onPickFiles = (files: FileList | null) => {
    if (!files) return
    Array.from(files).forEach((f) => upload.mutate(f))
    if (fileInput.current) fileInput.current.value = ''
  }

  const onNodeClick = (n: GraphNode) => {
    setAutoOrbit(false)
    setSelectedId(n.id)
    setFocusedId(n.id)
  }

  const onNodeRightClick = (n: GraphNode) => {
    setAutoOrbit(false)
    setSelectedId(null)
    setPathSeedA(n.name)
    setPathKey((k) => k + 1)
    setView('contradictions')
  }

  const handleHighlightGraph = (
    nodeNames: string[],
    edgePairs: string[],
    label: string
  ) => {
    setHighlightedNodeNames(new Set(nodeNames.map((n) => n.toLowerCase())))
    setHighlightedEdgePairs(new Set(edgePairs.map((p) => p.toLowerCase())))
    setSpotlightLabel(label)
    if (nodeNames.length > 0) {
      const target = nodes.find(
        (n) => n.name.toLowerCase() === nodeNames[0].toLowerCase()
      )
      if (target) {
        setAutoOrbit(false)
        setFocusedId(target.id)
      }
    }
  }

  const clearSpotlight = () => {
    setHighlightedNodeNames(new Set())
    setHighlightedEdgePairs(new Set())
    setSpotlightLabel(null)
    setEgoNodeId(null)
    setSelectedDocId('all')
  }

  const focusNodeByName = (name: string) => {
    const match = nodes.find(
      (n) =>
        n.name.toLowerCase() === name.toLowerCase() ||
        (n.aliases || []).some((a) => a.toLowerCase() === name.toLowerCase())
    )
    if (match) {
      setAutoOrbit(false)
      setSelectedId(match.id)
      setFocusedId(match.id)
    }
  }

  const doneDocs = docs.filter((d) => d.status === 'done')
  const totalChunks = doneDocs.reduce((a, d) => a + d.chunk_count, 0)

  const llmLabel = llmQ.data
    ? llmQ.data.mode === 'offline'
      ? 'Ollama'
      : `Cloud (${llmQ.data.cloud.order[0]?.toUpperCase()})`
    : 'LLM'

  const activeBannerLabel =
    spotlightLabel ||
    (egoNodeId
      ? `2-Hop Isolation: ${selectedNode?.name ?? 'Node'}`
      : selectedDocId !== 'all'
      ? `Paper: ${docs.find((d) => d.id === selectedDocId)?.title ?? 'Filtered'}`
      : null)

  return (
    <div className="relative w-screen h-screen bg-ethereal-space overflow-hidden">
      <input
        ref={fileInput}
        type="file"
        accept={ACCEPT}
        multiple
        hidden
        onChange={(e) => onPickFiles(e.target.files)}
      />

      <TopNav
        activeView={view}
        setActiveView={setView}
        searchQuery={search}
        setSearchQuery={setSearch}
        onSelectSearchResult={(nodeId) => {
          setAutoOrbit(false)
          setSelectedId(nodeId)
          setFocusedId(nodeId)
          setView('graph')
        }}
        nodes={nodes}
        backendHealthy={health.data?.ok === true}
        rerankerStatus={health.data?.reranker ?? 'offline'}
        llmLabel={llmLabel}
        onOpenLLMModal={() => setLlmModalOpen(true)}
        nodeCount={statsQ.data?.node_count ?? nodes.length}
        edgeCount={statsQ.data?.edge_count ?? edges.length}
        communityCount={communities.length}
        contradictionCount={tensionCount}
        onUploadClick={() => fileInput.current?.click()}
      />

      {health.data && health.data.ok === false && (
        <div className="fixed top-14 left-1/2 -translate-x-1/2 z-50 px-3.5 py-1.5 rounded-xl bg-[#d9534f]/15 border border-[#d9534f]/40 text-xs text-[#f08a86] flex items-center gap-2">
          <AlertTriangle className="w-3.5 h-3.5" />
          Cannot reach the Ariadne API on port 8100.
        </div>
      )}

      {toast && (
        <div
          onClick={() => setToast(null)}
          className={`fixed top-14 right-4 z-50 px-3.5 py-2 rounded-xl text-xs font-mono cursor-pointer border shadow-2xl ${
            toast.kind === 'ok'
              ? 'bg-[#5ec2a8]/15 border-[#5ec2a8]/50 text-[#8ee3cb]'
              : 'bg-[#d9534f]/15 border-[#d9534f]/50 text-[#f08a86]'
          }`}
        >
          {toast.text}
        </div>
      )}

      {/* 3D Knowledge Graph Canvas */}
      {nodes.length > 0 && (
        <Graph3D
          key={cameraKey}
          nodes={nodes}
          edges={edges}
          selectedNodeId={selectedId}
          focusedNodeId={focusedId}
          onNodeClick={onNodeClick}
          onNodeRightClick={onNodeRightClick}
          onNodeHover={(n) => setHoveredNode(n)}
          activeRelationshipCategories={relCategories}
          activeNodeTypes={nodeTypes}
          selectedCluster={cluster}
          selectedDocId={selectedDocId}
          searchQuery={search}
          autoOrbit={autoOrbit}
          onStopAutoOrbit={() => setAutoOrbit(false)}
          livingMotion={livingMotion}
          formation={formation}
          repulsion={repulsion}
          maxNodes={maxNodes}
          labelMode={labelMode}
          highlightedNodeNames={highlightedNodeNames}
          highlightedEdgePairs={highlightedEdgePairs}
          egoNodeId={egoNodeId}
        />
      )}

      {view === 'graph' && graphQ.isLoading && (
        <div className="absolute inset-0 flex items-center justify-center text-zinc-400 gap-2 z-20 pointer-events-none">
          <Loader2 className="w-5 h-5 animate-spin text-[#e5b869]" />
          <span className="font-mono text-xs">Loading 3D Knowledge Graph…</span>
        </div>
      )}

      {view === 'graph' && graphQ.isSuccess && nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center px-6 z-20">
          <div className="glass-panel rounded-3xl p-8 max-w-lg text-center">
            <Network className="w-8 h-8 mx-auto mb-3 text-[#e5b869]" />
            <h2 className="font-brand text-lg text-[#f3efe6] mb-2">No knowledge graph yet</h2>
            <p className="text-xs text-zinc-400 leading-relaxed">
              {doneDocs.length > 0
                ? `${doneDocs.length} document(s) are indexed (${totalChunks} passages). `
                : 'Upload a document to get started. '}
              Once entities and relationships are extracted, your 3D graph will appear here.
            </p>
            <div className="mt-5 flex justify-center gap-3">
              <button
                onClick={() => setView('query')}
                className="px-4 py-2 rounded-full bg-gradient-to-r from-[#e5b869] to-[#c8963e] text-[#080a0a] text-xs font-semibold cursor-pointer shadow-[0_0_20px_rgba(229,184,105,0.28)]"
              >
                Ask a question
              </button>
              <button
                onClick={() => setView('vault')}
                className="px-4 py-2 rounded-full glass-panel text-zinc-200 text-xs cursor-pointer"
              >
                Inspect Vault
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Compact Top-Left Hub Pill & Hover Badge (starts collapsed so view is clean) */}
      {view === 'graph' && nodes.length > 0 && (
        <NeuralTelemetryHUD
          nodes={nodes}
          hoveredNode={hoveredNode}
          selectedNodeId={selectedId}
          onSelectNode={(n) => {
            setAutoOrbit(false)
            setSelectedId(n.id)
            setFocusedId(n.id)
          }}
          onOpenPathFinder={(name) => {
            setSelectedId(null)
            setPathSeedA(name)
            setPathKey((k) => k + 1)
            setView('contradictions')
          }}
        />
      )}

      {/* Overlay Views */}
      {view === 'query' && (
        <QueryView
          key={`${askSeed}-${askKey}`}
          initialQuestion={askSeed}
          hasDocuments={doneDocs.length > 0}
          docs={docs}
          nodes={nodes}
          onHighlightGraph={handleHighlightGraph}
          onFocusNodeByName={(name) => {
            focusNodeByName(name)
            setView('graph')
          }}
          onSwitchToPure3D={() => setView('graph')}
        />
      )}

      {view === 'contradictions' && (
        <ContradictionsView
          key={`${pathSeedA}-${pathKey}`}
          nodes={nodes}
          edges={edges}
          initialEntityA={pathSeedA}
          onHighlightGraph={handleHighlightGraph}
          onFocusNodeByName={(name) => {
            focusNodeByName(name)
            setView('graph')
          }}
          onSwitchToPure3D={() => setView('graph')}
        />
      )}

      {view === 'communities' && (
        <CommunitiesView
          nodes={nodes}
          onHighlightGraph={handleHighlightGraph}
          onAskAboutTheme={(q) => {
            setSelectedId(null)
            setAskSeed(q)
            setAskKey((k) => k + 1)
            setView('query')
          }}
          onSwitchToPure3D={() => setView('graph')}
        />
      )}

      {view === 'vault' && (
        <VaultView
          docs={docs}
          jobStreams={jobStreams}
          onUploadClick={() => fileInput.current?.click()}
          onDropFiles={(files) => onPickFiles(files)}
          onFilter3DByDoc={(docId) => {
            setSelectedDocId(docId)
            setFormation('pillars')
            setView('graph')
          }}
          onAskDoc={(docTitle) => {
            setSelectedId(null)
            setAskSeed(`What are the core contributions and findings in ${docTitle}?`)
            setAskKey((k) => k + 1)
            setView('query')
          }}
          onSwitchToPure3D={() => setView('graph')}
        />
      )}

      {/* Right-Side Entity Drawer (only in 3D Graph view when a node is clicked) */}
      {view === 'graph' && (
        <ConceptDrawer
          node={selectedNode}
          allNodes={nodes}
          egoNodeId={egoNodeId}
          onToggleEgo={(id) => setEgoNodeId(id)}
          onClose={() => {
            setSelectedId(null)
            setFocusedId(null)
          }}
          onSelectNode={(id) => {
            setAutoOrbit(false)
            setSelectedId(id)
            setFocusedId(id)
          }}
          onAskAboutNode={(concept) => {
            setSelectedId(null)
            setAskSeed(`What do my documents say about ${concept}?`)
            setAskKey((k) => k + 1)
            setView('query')
          }}
          onStartPathFromNode={(concept) => {
            setSelectedId(null)
            setPathSeedA(concept)
            setPathKey((k) => k + 1)
            setView('contradictions')
          }}
        />
      )}

      {/* Compact Bottom 3D Dock */}
      {view === 'graph' && nodes.length > 0 && (
        <FloatingControls
          activeRelationshipCategories={relCategories}
          setActiveRelationshipCategories={setRelCategories}
          activeNodeTypes={nodeTypes}
          setActiveNodeTypes={setNodeTypes}
          autoOrbit={autoOrbit}
          setAutoOrbit={setAutoOrbit}
          livingMotion={livingMotion}
          setLivingMotion={setLivingMotion}
          formation={formation}
          setFormation={setFormation}
          repulsion={repulsion}
          setRepulsion={setRepulsion}
          maxNodes={maxNodes}
          setMaxNodes={setMaxNodes}
          totalNodesCount={nodes.length}
          labelMode={labelMode}
          setLabelMode={setLabelMode}
          onResetCamera={() => {
            setAutoOrbit(false)
            setFocusedId(null)
            setCameraKey((k) => k + 1)
          }}
          selectedCluster={cluster}
          setSelectedCluster={setCluster}
          clusters={clusters}
          selectedDocId={selectedDocId}
          setSelectedDocId={setSelectedDocId}
          docs={doneDocs}
          activeSpotlightLabel={activeBannerLabel}
          onClearSpotlight={clearSpotlight}
        />
      )}

      <LLMConfigModal open={llmModalOpen} onClose={() => setLlmModalOpen(false)} />
    </div>
  )
}
