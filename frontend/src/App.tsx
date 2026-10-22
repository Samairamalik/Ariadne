import { useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Loader2, Network, AlertTriangle } from 'lucide-react'
import { documentsApi, graphApi, queryApi } from './api/client'
import { TopNav } from './components/TopNav'
import { Graph3D } from './components/Graph3D'
import { FloatingControls } from './components/FloatingControls'
import { ConceptDrawer } from './components/ConceptDrawer'
import { QueryView } from './views/QueryView'
import { VaultView } from './views/VaultView'
import { ContradictionsView } from './views/ContradictionsView'
import type { GraphNode } from './types'

type View = 'graph' | 'vault' | 'query' | 'contradictions'

const ACCEPT = '.pdf,.pptx,.docx,.md,.markdown,.txt,.html,.htm'
const ALL_NODE_TYPES = ['concept', 'model', 'method', 'dataset', 'paper', 'person', 'other']
const BASE_RELATIONSHIPS = ['BUILDS_ON', 'SUPPORTS', 'CONTRADICTS', 'REQUIRED_BY', 'TANGENTIALLY_RELATED']

export default function App() {
  const qc = useQueryClient()
  const fileInput = useRef<HTMLInputElement>(null)

  const [view, setView] = useState<View>('query')
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [focusedId, setFocusedId] = useState<string | null>(null)
  const [relTypes, setRelTypes] = useState<string[]>(BASE_RELATIONSHIPS)
  const [nodeTypes, setNodeTypes] = useState<string[]>(ALL_NODE_TYPES)
  const [cluster, setCluster] = useState('all')
  const [autoOrbit, setAutoOrbit] = useState(false)
  const [repulsion, setRepulsion] = useState(50)
  const [cameraKey, setCameraKey] = useState(0)
  const [askSeed, setAskSeed] = useState('')
  const [toast, setToast] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null)

  const health = useQuery({ queryKey: ['health'], queryFn: queryApi.checkHealth, refetchInterval: 5000 })
  const docsQ = useQuery({
    queryKey: ['documents'],
    queryFn: documentsApi.listDocuments,
    // poll quickly while anything is still being ingested
    refetchInterval: (q) => (q.state.data?.some((d) => !['done', 'failed'].includes(d.status)) ? 1500 : 10000),
  })
  const graphQ = useQuery({ queryKey: ['graph'], queryFn: graphApi.getGraph, retry: false })
  const statsQ = useQuery({ queryKey: ['stats'], queryFn: graphApi.getStats, retry: false, refetchInterval: 15000 })

  const docs = docsQ.data ?? []
  const nodes = graphQ.data?.nodes ?? []
  const edges = graphQ.data?.edges ?? []
  const selectedNode = useMemo(() => nodes.find((n) => n.id === selectedId) ?? null, [nodes, selectedId])
  const clusters = useMemo(() => Array.from(new Set(nodes.map((n) => n.cluster))).sort(), [nodes])
  const allRelTypes = useMemo(() => Array.from(new Set([...BASE_RELATIONSHIPS, ...edges.map((e) => e.predicate)])), [edges])
  const activeRelTypes = useMemo(() => Array.from(new Set([...relTypes, ...allRelTypes.filter((r) => !BASE_RELATIONSHIPS.includes(r))])), [relTypes, allRelTypes])

  const upload = useMutation({
    mutationFn: (file: File) => documentsApi.uploadDocument(file),
    onSuccess: (r, file) => {
      setToast({ kind: 'ok', text: r.duplicate ? `"${file.name}" is already indexed.` : `Uploading "${file.name}" — processing started.` })
      qc.invalidateQueries({ queryKey: ['documents'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
      setView('vault')
    },
    onError: (e: unknown) => {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setToast({ kind: 'err', text: detail ?? (e as Error).message ?? 'Upload failed' })
    },
  })

  const onPickFiles = (files: FileList | null) => {
    if (!files) return
    Array.from(files).forEach((f) => upload.mutate(f))
    if (fileInput.current) fileInput.current.value = ''
  }

  const onNodeClick = (n: GraphNode) => {
    setSelectedId(n.id)
    setFocusedId(n.id)
  }

  const doneDocs = docs.filter((d) => d.status === 'done')
  const totalChunks = doneDocs.reduce((a, d) => a + d.chunk_count, 0)

  return (
    <div className="relative w-screen h-screen bg-ethereal-space overflow-hidden">
      <input ref={fileInput} type="file" accept={ACCEPT} multiple hidden onChange={(e) => onPickFiles(e.target.files)} />

      <TopNav
        activeView={view}
        setActiveView={setView}
        searchQuery={search}
        setSearchQuery={setSearch}
        backendHealthy={health.data?.ok === true}
        rerankerStatus={health.data?.reranker ?? 'offline'}
        nodeCount={statsQ.data?.node_count ?? nodes.length}
        edgeCount={statsQ.data?.edge_count ?? edges.length}
        contradictionCount={statsQ.data?.contradiction_count ?? 0}
        onUploadClick={() => fileInput.current?.click()}
      />

      {health.data && health.data.ok === false && (
        <div className="fixed top-20 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-lg bg-[#ff1744]/15 border border-[#ff1744]/40 text-sm text-[#ff8aa1] flex items-center gap-2">
          <AlertTriangle className="w-4 h-4" />
          Cannot reach the Ariadne API. Start it with: python -m uvicorn app.main:app --port 8100
        </div>
      )}

      {toast && (
        <div
          onClick={() => setToast(null)}
          className={`fixed bottom-6 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-lg text-sm cursor-pointer border ${
            toast.kind === 'ok' ? 'bg-[#00ff88]/10 border-[#00ff88]/40 text-[#7dffbd]' : 'bg-[#ff1744]/15 border-[#ff1744]/40 text-[#ff8aa1]'
          }`}
        >
          {toast.text}
        </div>
      )}

      {view === 'query' && <QueryView key={askSeed} initialQuestion={askSeed} hasDocuments={doneDocs.length > 0} />}
      {view === 'vault' && <VaultView docs={docs} onUploadClick={() => fileInput.current?.click()} />}
      {view === 'contradictions' && <ContradictionsView nodes={nodes} edges={edges} />}

      {view === 'graph' && (
        <>
          {graphQ.isLoading && (
            <div className="absolute inset-0 flex items-center justify-center text-zinc-400 gap-2">
              <Loader2 className="w-5 h-5 animate-spin" /> Loading graph…
            </div>
          )}
          {graphQ.isError && (
            <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-[#ff8aa1]">
              Couldn't load the graph from the API. Check that the backend is running.
            </div>
          )}
          {graphQ.isSuccess && nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center px-6">
              <div className="glass-panel rounded-2xl p-8 max-w-lg text-center">
                <Network className="w-8 h-8 mx-auto mb-3 text-[#00f0ff]" />
                <h2 className="font-headline text-lg text-white mb-2">No knowledge graph yet</h2>
                <p className="text-sm text-zinc-400 leading-relaxed">
                  {doneDocs.length > 0
                    ? `${doneDocs.length} document(s) are indexed (${totalChunks} passages), so search and cited answers work now. `
                    : 'Upload a document to get started. '}
                  Extracting concepts and relationships into a graph is the next build step; until then this view
                  stays empty rather than showing made-up connections.
                </p>
                <div className="mt-5 flex justify-center gap-3">
                  <button onClick={() => setView('query')} className="px-4 py-2 rounded-full bg-gradient-to-r from-[#ff2d9b] to-[#00f0ff] text-white text-xs font-medium cursor-pointer">
                    Ask a question
                  </button>
                  <button onClick={() => setView('vault')} className="px-4 py-2 rounded-full glass-panel text-zinc-200 text-xs cursor-pointer">
                    Inspect documents
                  </button>
                </div>
              </div>
            </div>
          )}
          {nodes.length > 0 && (
            <>
              <Graph3D
                key={cameraKey}
                nodes={nodes}
                edges={edges}
                selectedNodeId={selectedId}
                hoveredNodeId={hoveredId}
                focusedNodeId={focusedId}
                onNodeClick={onNodeClick}
                onNodeHover={(n) => setHoveredId(n?.id ?? null)}
                activeRelationshipTypes={activeRelTypes}
                activeNodeTypes={nodeTypes}
                selectedCluster={cluster}
                searchQuery={search}
                autoOrbit={autoOrbit}
                repulsion={repulsion}
              />
              <FloatingControls
                activeRelationshipTypes={relTypes}
                setActiveRelationshipTypes={setRelTypes}
                activeNodeTypes={nodeTypes}
                setActiveNodeTypes={setNodeTypes}
                autoOrbit={autoOrbit}
                setAutoOrbit={setAutoOrbit}
                repulsion={repulsion}
                setRepulsion={setRepulsion}
                onResetCamera={() => {
                  setFocusedId(null)
                  setCameraKey((k) => k + 1)
                }}
                selectedCluster={cluster}
                setSelectedCluster={setCluster}
                clusters={clusters}
              />
              <ConceptDrawer
                node={selectedNode}
                onClose={() => {
                  setSelectedId(null)
                  setFocusedId(null)
                }}
                onSelectNode={(id) => {
                  setSelectedId(id)
                  setFocusedId(id)
                }}
                onAskAboutNode={(concept) => {
                  setAskSeed(`What do my documents say about ${concept}?`)
                  setView('query')
                }}
              />
            </>
          )}
        </>
      )}
    </div>
  )
}
