import axios from 'axios'
import type {
  DocumentItem,
  ChunkItem,
  GraphNode,
  GraphEdge,
  QueryResponse,
  SearchOnlyResponse,
  GraphLinkResponse,
  NodeDetailResponse,
  EntitySearchItem,
  EntityProfileResponse,
  EntityPathResponse,
  EntityAggregateResponse,
  CommunityItem,
  LLMStatusResponse,
  LLMTestResponse,
  IngestionJobProgress,
} from '../types'

const API_BASE = '/api'

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
})

// Safe fallback graph in case backend is starting with VITE_DEMO_GRAPH=1
export const FALLBACK_GRAPH: { nodes: GraphNode[]; edges: GraphEdge[] } = {
  nodes: [
    {
      id: 'n1',
      name: 'Attention Mechanism',
      label: 'Attention Mechanism',
      type: 'concept',
      cluster: 'Deep Learning',
      description: 'Dynamic sequence-weighting mechanism enabling direct cross-token dependency computation without recurrent bottlenecks.',
      aliases: ['Self-Attention', 'Scaled Dot-Product Attention'],
      mention_count: 32,
      degree: 6,
      confidence: 0.98,
      size: 18,
    },
    {
      id: 'n2',
      name: 'BERT',
      label: 'BERT',
      type: 'model',
      cluster: 'Language Models',
      description: 'Bidirectional Encoder Representations from Transformers pre-trained on masked language modeling and next sentence prediction.',
      aliases: ['BERT-Base', 'BERT-Large'],
      mention_count: 45,
      degree: 7,
      confidence: 0.99,
      size: 22,
    },
    {
      id: 'n3',
      name: 'Transformer Architecture',
      label: 'Transformer Architecture',
      type: 'model',
      cluster: 'Deep Learning',
      description: 'Foundational multi-head self-attention encoder-decoder network discarding sequential recurrence.',
      aliases: ['Vaswani Architecture'],
      mention_count: 38,
      degree: 5,
      confidence: 0.97,
      size: 19,
    },
    {
      id: 'n4',
      name: 'Masked Language Modeling',
      label: 'Masked Language Modeling',
      type: 'method',
      cluster: 'Pre-Training',
      description: 'Bidirectional token prediction cloze task where 15% of tokens are randomly masked to avoid directional bias.',
      aliases: ['MLM'],
      mention_count: 24,
      degree: 4,
      confidence: 0.96,
      size: 14,
    },
  ],
  edges: [
    {
      id: 'e1',
      source: 'n2',
      target: 'n3',
      type: 'BUILDS_ON',
      predicate: 'BUILDS_ON',
      evidence: 'BERT uses the multi-layer bidirectional Transformer encoder architecture.',
      confidence: 'high',
    },
    {
      id: 'e2',
      source: 'n2',
      target: 'n1',
      type: 'BUILDS_ON',
      predicate: 'BUILDS_ON',
      evidence: 'BERT applies multi-head self-attention across every layer.',
      confidence: 'high',
    },
    {
      id: 'e3',
      source: 'n2',
      target: 'n4',
      type: 'REQUIRED_BY',
      predicate: 'REQUIRED_BY',
      evidence: 'Pre-training BERT requires the masked language modeling objective.',
      confidence: 'high',
    },
  ],
}

export const graphApi = {
  async getGraph(): Promise<{ nodes: GraphNode[]; edges: GraphEdge[] }> {
    const res = await api.get('/graph')
    const nodes: GraphNode[] = res.data?.nodes ?? []
    const edges: GraphEdge[] = res.data?.edges ?? res.data?.links ?? []
    if (nodes.length === 0 && import.meta.env.VITE_DEMO_GRAPH === '1') return FALLBACK_GRAPH
    return { nodes, edges }
  },

  async getNodeDetail(nodeId: string): Promise<NodeDetailResponse | null> {
    try {
      const res = await api.get(`/graph/nodes/${nodeId}`)
      return res.data
    } catch {
      return null
    }
  },

  async getStats(): Promise<{ node_count: number; edge_count: number; document_count: number; contradiction_count: number }> {
    const res = await api.get('/graph/stats')
    return res.data
  },

  async searchEntities(name: string, limit = 8): Promise<EntitySearchItem[]> {
    if (!name || name.trim().length < 2) return []
    const res = await api.get('/graph/search', { params: { name: name.trim(), limit } })
    return res.data ?? []
  },

  async getEntityProfile(name: string): Promise<EntityProfileResponse | null> {
    try {
      const res = await api.get('/graph/entity', { params: { name: name.trim() } })
      return res.data
    } catch {
      return null
    }
  },

  async findPath(a: string, b: string, maxHops = 3): Promise<EntityPathResponse> {
    const res = await api.get('/graph/path', {
      params: { a: a.trim(), b: b.trim(), max_hops: maxHops },
    })
    return res.data
  },

  async aggregateEntity(entity: string, predicate?: string): Promise<EntityAggregateResponse | null> {
    try {
      const res = await api.get('/graph/aggregate', {
        params: { entity: entity.trim(), ...(predicate ? { predicate: predicate.trim() } : {}) },
      })
      return res.data
    } catch {
      return null
    }
  },

  async getCommunities(): Promise<CommunityItem[]> {
    const res = await api.get('/graph/communities')
    return res.data ?? []
  },

  async buildCommunities(): Promise<{ communities_found: number; built: number }> {
    const res = await api.post('/graph/communities/build')
    return res.data
  },
}

export const documentsApi = {
  async listDocuments(): Promise<DocumentItem[]> {
    const res = await api.get('/documents')
    return res.data || []
  },

  async getChunks(docId: string, limit = 200, offset = 0): Promise<{ stats: any; chunks: ChunkItem[] }> {
    const res = await api.get(`/documents/${docId}/chunks`, { params: { limit, offset } })
    return res.data
  },

  async deleteDocument(docId: string): Promise<void> {
    await api.delete(`/documents/${docId}`)
  },

  async uploadDocument(file: File): Promise<{ document_id: string; job_id: string; duplicate: boolean }> {
    const form = new FormData()
    form.append('file', file)
    const res = await api.post('/documents/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return res.data
  },

  streamJobStatus(
    jobId: string,
    onProgress: (progress: IngestionJobProgress) => void,
    onDone?: () => void
  ): () => void {
    const es = new EventSource(`${API_BASE}/jobs/${jobId}/status`)
    es.onmessage = (event) => {
      try {
        const data: IngestionJobProgress = JSON.parse(event.data)
        onProgress(data)
        if (data.status === 'done' || data.status === 'failed') {
          es.close()
          onDone?.()
        }
      } catch {
        // ignore malformed frame
      }
    }
    es.onerror = () => {
      es.close()
      onDone?.()
    }
    return () => es.close()
  },
}

export const queryApi = {
  async queryRAG(params: {
    question: string
    document_ids?: string[] | null
    top_k?: number
    generate?: boolean
    rerank?: boolean
    use_graph?: boolean
    verify_claims?: boolean
    general_knowledge?: boolean
  }): Promise<QueryResponse> {
    const res = await api.post('/query', {
      question: params.question,
      document_ids: params.document_ids && params.document_ids.length > 0 ? params.document_ids : null,
      top_k: params.top_k ?? 6,
      generate: params.generate ?? true,
      rerank: params.rerank ?? true,
      use_graph: params.use_graph ?? true,
      verify_claims: params.verify_claims ?? false,
      general_knowledge: params.general_knowledge ?? true,
    })
    return res.data
  },

  async searchChunks(params: {
    question: string
    document_ids?: string[] | null
    top_k?: number
    rerank?: boolean
  }): Promise<SearchOnlyResponse> {
    const res = await api.post('/search', {
      question: params.question,
      document_ids: params.document_ids && params.document_ids.length > 0 ? params.document_ids : null,
      top_k: params.top_k ?? 6,
      rerank: params.rerank ?? true,
    })
    return res.data
  },

  async linkEntities(params: {
    question: string
    document_ids?: string[] | null
    top_k?: number
    rerank?: boolean
  }): Promise<GraphLinkResponse> {
    const res = await api.post('/graph/link', {
      question: params.question,
      document_ids: params.document_ids && params.document_ids.length > 0 ? params.document_ids : null,
      top_k: params.top_k ?? 6,
      rerank: params.rerank ?? true,
    })
    return res.data
  },

  async checkHealth(): Promise<{ ok: boolean; reranker: string }> {
    try {
      const res = await api.get('/health')
      return res.data
    } catch {
      return { ok: false, reranker: 'offline' }
    }
  },
}

export const llmApi = {
  async getStatus(): Promise<LLMStatusResponse> {
    const res = await api.get('/llm')
    return res.data
  },

  async configure(cfg: {
    mode?: 'offline' | 'cloud'
    ollama_model?: string
    cloud_order?: string[]
  }): Promise<LLMStatusResponse> {
    const res = await api.put('/llm', cfg)
    return res.data
  },

  async testConnection(): Promise<LLMTestResponse> {
    const res = await api.post('/llm/test')
    return res.data
  },
}
