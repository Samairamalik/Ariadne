export type NodeType = 'concept' | 'model' | 'method' | 'dataset' | 'paper' | 'person' | 'other'

export type RelationshipType =
  | 'BUILDS_ON'
  | 'CONTRADICTS'
  | 'SUPPORTS'
  | 'IS_EXAMPLE_OF'
  | 'REQUIRED_BY'
  | 'TANGENTIALLY_RELATED'
  | string

export type SpatialFormation = 'neural' | 'galaxy' | 'sphere' | 'pillars'

export interface GraphNode {
  id: string
  name: string
  canonical_name?: string
  label?: string
  type: NodeType
  cluster: string
  description: string
  aliases?: string[]
  mention_count: number
  degree: number
  confidence: number
  document_id?: string | null
  doc_title?: string | null
  size?: number
  visualSize?: number
  x?: number
  y?: number
  z?: number
  fx?: number
  fy?: number
  fz?: number
  color?: string
}

export interface GraphEdge {
  id: string
  source: string | GraphNode
  target: string | GraphNode
  source_id?: string
  target_id?: string
  type: RelationshipType
  predicate: RelationshipType
  evidence: string
  confidence?: 'high' | 'medium' | 'low' | string
  document_id?: string | null
  doc_title?: string | null
}

export interface DocumentItem {
  id: string
  filename: string
  title: string | null
  author: string | null
  doc_type: 'pdf' | 'markdown' | 'text' | 'pptx' | 'docx' | 'html'
  page_count: number | null
  token_count: number | null
  status: 'pending' | 'chunking' | 'embedding' | 'extracting' | 'resolving' | 'done' | 'failed'
  error: string | null
  ingested_at: string
  chunk_count: number
}

export interface IngestionJobProgress {
  stage: string
  chunks_total: number | null
  chunks_done: number | null
  status: string
  error: string | null
}

export interface ChunkItem {
  id: string
  chunk_index: number
  kind: 'prose' | 'table' | 'code' | 'footnote' | 'caption'
  section_path: string | null
  page_start: number | null
  page_end: number | null
  token_count: number
  text: string
}

export interface RetrievalSource {
  n: number
  chunk_id: string
  document_id: string
  doc_title: string | null
  section_path: string | null
  pages: [number | null, number | null]
  text: string
  scores: {
    cosine: number
    rrf: number
    rerank: number | null
  }
  found_by: Record<string, number>
}

export interface NLIClaimCheck {
  sentence: string
  cited: number[]
  entailment: number
  contradiction: number
  verdict: 'supported' | 'unverified' | 'contradicted'
  conflict: boolean
}

export interface NLIVerification {
  enabled?: boolean
  available?: boolean
  claims?: number
  supported?: number
  unverified?: number
  contradicted?: number
  conflicting?: number
  checks?: NLIClaimCheck[]
}

export interface EvidenceTrailStep {
  hop: number
  fact: string
  quote: string
  document: string
  section: string
  source: number
}

export interface GraphFact {
  subject: string
  predicate: string
  object: string
  hop: number
}

export interface QueryResponse {
  answer: string | null
  llm_error?: string | null
  llm_provider?: string | null
  llm_model?: string | null
  refused: boolean | null
  relevant?: boolean
  source?: 'documents' | 'general_knowledge' | 'none'
  cited: (string | number)[]
  invalid_citations: (string | number)[]
  uncited_sentences: string[]
  verification?: NLIVerification
  sources: RetrievalSource[]
  path_taken: 'graph' | 'hybrid' | 'chunk' | string
  intent?: string
  evidence_trail?: EvidenceTrailStep[]
  structured?: Record<string, any> | null
  question_parts?: string[]
  overview_used?: string[]
  graph?: {
    seeds: string[]
    hops: number
    extra_chunks: number
    extras: Array<{ doc: string; section: string; rerank: number | null }>
    facts: GraphFact[]
  }
  timing: {
    total_retrieval_ms?: number
    total_ms: number
    candidates: number
    reranked: boolean
    targeted_docs?: number
    focus?: Record<string, string>
    parts?: string[]
  }
}

export interface SearchOnlyResponse {
  results: RetrievalSource[]
  relevant: boolean
  timing: {
    total_retrieval_ms: number
    candidates: number
    reranked: boolean
    targeted_docs?: number
    focus?: Record<string, string>
    parts?: string[]
  }
}

export interface GraphLinkNode {
  id: string
  name: string
  type: NodeType
  score: number
  found_by: Record<string, any>
}

export interface GraphLinkResponse {
  relevant: boolean
  nodes: GraphLinkNode[]
}

export interface NodeDetailResponse {
  node: {
    id: string
    name: string
    aliases: string[]
    type: NodeType
    description: string
    mention_count: number
    first_seen_doc: string | null
    doc_title: string | null
    confidence: number
  }
  relationships: Array<{
    id: string
    source_id: string
    target_id: string
    source_name: string
    target_name: string
    predicate: string
    evidence_text: string
    confidence: string
    doc_title: string | null
  }>
  evidence_chunks: Array<{
    id: string
    document_id: string
    chunk_index: number
    section_path: string | null
    page: number | null
    text: string
    doc_title: string | null
  }>
}

export interface EntitySearchItem {
  id: string
  name: string
  type: NodeType
  papers: number
  mentions: number
  match: number
}

export interface EntityFactItem {
  subject: string
  predicate: string
  object: string
  quote: string
  document: string
  document_id: string
  section: string
  page: number | null
  chunk_id: string | null
  direction?: 'in' | 'out' | 'forward' | 'reverse'
}

export interface EntityProfileResponse {
  entity: {
    id: string
    name: string
    type: NodeType
    aliases: string[]
    documents: number
    facts: number
  }
  documents: Array<{
    document: string
    document_id: string
    filename: string
    facts: EntityFactItem[]
  }>
  predicates: Record<string, number>
  related: Array<{
    name: string
    facts: number
  }>
}

export interface PathStepItem extends EntityFactItem {
  from: string
  to: string
  direction: 'forward' | 'reverse'
}

export interface EntityPathResponse {
  from: string
  to: string
  max_hops: number
  connected: boolean
  paths: Array<{
    hops: number
    steps: PathStepItem[]
  }>
}

export interface EntityAggregateResponse {
  entity: string
  predicate: string | null
  papers: number
  documents: Array<{
    document: string
    document_id: string
    facts: EntityFactItem[]
    chunk_id: string | null
    mentioned?: boolean
  }>
}

export interface CommunityItem {
  title: string
  summary: string
  size: number
}

export interface LLMStatusResponse {
  mode: 'offline' | 'cloud'
  will_try: string[]
  sends_text_off_machine: boolean
  ollama: {
    selected: string
    installed: string[]
    error: string | null
  }
  cloud: {
    order: string[]
    allow_local_fallback: boolean
    providers: Record<
      string,
      {
        configured: boolean
        model: string
      }
    >
  }
}

export interface LLMTestResponse {
  ok: boolean
  provider?: string
  model?: string
  reply?: string
  latency_ms?: number
  error?: string
}
