import json
import logging
from uuid import UUID, uuid4
from fastapi import APIRouter, HTTPException
from app import db
from app.config import settings

log = logging.getLogger("ariadne.graph")
router = APIRouter(prefix="/api/graph", tags=["graph"])

# High-fidelity concept seeds derived from Ariadne's domain documents (BERT & Vision Dehazing)
DEFAULT_SEEDS = [
    {
        "name": "Attention Mechanism",
        "type": "concept",
        "cluster": "Deep Learning Architectures",
        "description": "Dynamic weighting mechanism enabling sequence models to correlate all token representations irrespective of distance.",
        "confidence": 0.98,
        "mention_count": 28,
        "aliases": ["Self-Attention", "Multi-Head Attention", "Scaled Dot-Product Attention"]
    },
    {
        "name": "BERT",
        "type": "model",
        "cluster": "Language Models",
        "description": "Bidirectional Encoder Representations from Transformers designed to pre-train deep bidirectional representations.",
        "confidence": 0.99,
        "mention_count": 42,
        "aliases": ["Bidirectional Encoder Representations", "BERT-Base", "BERT-Large"]
    },
    {
        "name": "Transformer Architecture",
        "type": "model",
        "cluster": "Deep Learning Architectures",
        "description": "Feed-forward neural architecture based entirely on attention mechanisms dispensing with recurrence and convolutions.",
        "confidence": 0.97,
        "mention_count": 35,
        "aliases": ["Vaswani Transformer", "Encoder-Decoder Transformer"]
    },
    {
        "name": "Masked Language Modeling",
        "type": "method",
        "cluster": "Self-Supervised Learning",
        "description": "Cloze task pre-training objective where random input tokens are masked and predicted from bidirectional context.",
        "confidence": 0.96,
        "mention_count": 22,
        "aliases": ["MLM", "Cloze Task"]
    },
    {
        "name": "Next Sentence Prediction",
        "type": "method",
        "cluster": "Self-Supervised Learning",
        "description": "Binarized pre-training objective predicting whether segment B follows segment A in the original corpus.",
        "confidence": 0.91,
        "mention_count": 15,
        "aliases": ["NSP", "Sentence Pair Classification"]
    },
    {
        "name": "Fine-Tuning Paradigm",
        "type": "method",
        "cluster": "Transfer Learning",
        "description": "Adapting all pre-trained model parameters to downstream supervised tasks with minimal task-specific layers.",
        "confidence": 0.94,
        "mention_count": 19,
        "aliases": ["Pre-train then Fine-tune", "Downstream Transfer"]
    },
    {
        "name": "Dark Channel Prior",
        "type": "method",
        "cluster": "Computer Vision",
        "description": "Statistical prior based on low intensity pixels in non-sky regions, used for single-image haze removal.",
        "confidence": 0.92,
        "mention_count": 16,
        "aliases": ["DCP", "He Prior"]
    },
    {
        "name": "Scene-Adaptive Dehazing",
        "type": "method",
        "cluster": "Autonomous Driving Vision",
        "description": "Real-time atmospheric scattering compensation dynamically adjusting transmission maps in driving scenes.",
        "confidence": 0.95,
        "mention_count": 24,
        "aliases": ["Adaptive Transmission", "Fast Dehazing"]
    },
    {
        "name": "Atmospheric Scattering Model",
        "type": "concept",
        "cluster": "Physics & Optics",
        "description": "Physical formulation I(x) = J(x)t(x) + A(1 - t(x)) modelling hazy image formation with scene radiance J and airlight A.",
        "confidence": 0.95,
        "mention_count": 14,
        "aliases": ["Koschmieder Model", "Haze Formation Physics"]
    },
    {
        "name": "Real-Time Object Detection",
        "type": "method",
        "cluster": "Autonomous Driving Vision",
        "description": "High-framerate bounding box regression and classification operating under severe visibility degradation.",
        "confidence": 0.93,
        "mention_count": 18,
        "aliases": ["YOLO Driving", "Fast Detection"]
    },
    {
        "name": "YOLO Detection Network",
        "type": "model",
        "cluster": "Computer Vision",
        "description": "Single-stage convolutional neural detector predicting spatial coordinates and class probabilities simultaneously.",
        "confidence": 0.94,
        "mention_count": 12,
        "aliases": ["You Only Look Once", "One-Stage Detector"]
    },
    {
        "name": "GLUE Benchmark",
        "type": "dataset",
        "cluster": "Evaluation & Benchmarks",
        "description": "General Language Understanding Evaluation benchmark collection of diverse natural language understanding tasks.",
        "confidence": 0.96,
        "mention_count": 17,
        "aliases": ["General Language Understanding Evaluation", "GLUE"]
    },
    {
        "name": "SQuAD Benchmark",
        "type": "dataset",
        "cluster": "Evaluation & Benchmarks",
        "description": "Stanford Question Answering Dataset consisting of questions posed on Wikipedia articles with answer spans.",
        "confidence": 0.95,
        "mention_count": 16,
        "aliases": ["SQuAD v1.1", "SQuAD v2.0"]
    },
    {
        "name": "Reciprocal Rank Fusion",
        "type": "method",
        "cluster": "Information Retrieval",
        "description": "Unsupervised rank aggregation formula scoring candidates by 1 / (k + rank) across dense and sparse retrievers.",
        "confidence": 0.97,
        "mention_count": 11,
        "aliases": ["RRF", "Hybrid Fusion"]
    },
    {
        "name": "Cross-Encoder Reranker",
        "type": "model",
        "cluster": "Information Retrieval",
        "description": "Full-attention relevance scoring model computing joint question-passage representations to overcome bi-encoder loss.",
        "confidence": 0.96,
        "mention_count": 13,
        "aliases": ["MiniLM Reranker", "Cross-Encoder"]
    },
    {
        "name": "Longformer",
        "type": "model",
        "cluster": "Efficient Transformers",
        "description": "Sliding window attention mechanism with local windows and designated global tokens scaling linearly with length.",
        "confidence": 0.93,
        "mention_count": 9,
        "aliases": ["Sliding Window Attention"]
    },
    {
        "name": "Mamba State Space",
        "type": "model",
        "cluster": "Linear Time Models",
        "description": "Selective state space architecture computing linear-time sequential recurrence with hardware-aware scan.",
        "confidence": 0.92,
        "mention_count": 8,
        "aliases": ["Selective SSM", "S6"]
    }
]

DEFAULT_EDGES = [
    ("BERT", "Transformer Architecture", "BUILDS_ON", "BERT builds directly upon the multi-layer bidirectional Transformer encoder architecture.", "high"),
    ("BERT", "Attention Mechanism", "BUILDS_ON", "BERT utilizes multi-head self-attention to learn bi-directional contextual representations across all layers.", "high"),
    ("BERT", "Masked Language Modeling", "REQUIRED_BY", "Pre-training BERT requires the masked language model task to avoid cyclical label leakage.", "high"),
    ("BERT", "Next Sentence Prediction", "BUILDS_ON", "BERT couples MLM with a next sentence prediction task to model sentence relationships.", "medium"),
    ("BERT", "GLUE Benchmark", "SUPPORTS", "BERT obtains state-of-the-art results across 9 GLUE benchmark tasks.", "high"),
    ("BERT", "SQuAD Benchmark", "SUPPORTS", "BERT achieves dramatic improvements on SQuAD v1.1 and v2.0 question answering benchmarks.", "high"),
    ("BERT", "Fine-Tuning Paradigm", "SUPPORTS", "BERT establishes the pre-train and fine-tune paradigm as the dominant approach for NLU.", "high"),
    ("Scene-Adaptive Dehazing", "Atmospheric Scattering Model", "BUILDS_ON", "The dehazing algorithm is formulated upon the classical atmospheric scattering physical model.", "high"),
    ("Scene-Adaptive Dehazing", "Dark Channel Prior", "BUILDS_ON", "Refines the dark channel prior with boundary constraints and adaptive scene luminance compensation.", "high"),
    ("Scene-Adaptive Dehazing", "Real-Time Object Detection", "SUPPORTS", "Enhances visibility in foggy driving video streams, boosting downstream detection accuracy.", "high"),
    ("Real-Time Object Detection", "YOLO Detection Network", "BUILDS_ON", "Deploys an optimized YOLO model for low-latency obstacle identification in dynamic road environments.", "high"),
    ("Reciprocal Rank Fusion", "BERT", "TANGENTIALLY_RELATED", "Combines sparse lexical signals with dense semantic embeddings produced by Transformer encoders.", "medium"),
    ("Cross-Encoder Reranker", "Reciprocal Rank Fusion", "BUILDS_ON", "Reranks top candidates generated by reciprocal rank fusion using fine-grained token cross-attention.", "high"),
    ("Longformer", "Attention Mechanism", "BUILDS_ON", "Restricts full attention to a sliding window of 512 tokens to reduce O(n^2) computational overhead.", "high"),
    ("Mamba State Space", "Attention Mechanism", "CONTRADICTS", "Argues that recurrence with input-dependent selection surpasses attention scaling limitations without requiring an attention matrix.", "high"),
    ("Fine-Tuning Paradigm", "Mamba State Space", "TANGENTIALLY_RELATED", "Pre-trained state space backbones can also be fine-tuned on task-specific sequence inputs.", "medium"),
]


async def _seed_graph_if_empty(conn):
    count = await conn.fetchval("SELECT count(*) FROM nodes")
    if count > 0:
        return

    # Fetch document IDs if present
    docs = await conn.fetch("SELECT id, title FROM documents ORDER BY ingested_at ASC")
    bert_doc_id = docs[0]["id"] if len(docs) > 0 else None
    dehaze_doc_id = docs[1]["id"] if len(docs) > 1 else None

    name_to_id = {}
    for seed in DEFAULT_SEEDS:
        doc_id = bert_doc_id if "bert" in seed["name"].lower() or "language" in seed["cluster"].lower() else dehaze_doc_id
        node_id = uuid4()
        name_to_id[seed["name"]] = node_id
        await conn.execute(
            """INSERT INTO nodes (id, canonical_name, aliases, node_type, description, mention_count, first_seen_doc)
               VALUES ($1, $2, $3, $4, $5, $6, $7)
               ON CONFLICT (canonical_name, node_type) DO UPDATE SET mention_count = EXCLUDED.mention_count
            """,
            node_id, seed["name"], seed["aliases"], seed["type"], seed["description"], seed["mention_count"], doc_id
        )

    for src_name, tgt_name, predicate, evidence, conf in DEFAULT_EDGES:
        if src_name in name_to_id and tgt_name in name_to_id:
            edge_id = uuid4()
            doc_id = bert_doc_id if "bert" in src_name.lower() or "attention" in src_name.lower() else dehaze_doc_id
            await conn.execute(
                """INSERT INTO edges (id, source_id, target_id, predicate, evidence_text, confidence, document_id)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)
                   ON CONFLICT DO NOTHING
                """,
                edge_id, name_to_id[src_name], name_to_id[tgt_name], predicate, evidence, conf, doc_id
            )


@router.get("")
async def get_graph():
    """Retrieve all nodes and edges formatted for 3D force-directed layout."""
    async with db.pool().acquire() as conn:
        if settings.demo_graph:  # opt-in only; see config.demo_graph
            await _seed_graph_if_empty(conn)

        node_rows = await conn.fetch(
            """SELECT n.id, n.canonical_name, n.aliases, n.node_type, n.description,
                      n.mention_count, n.first_seen_doc, d.title AS doc_title,
                      (SELECT count(*) FROM edges e WHERE e.source_id = n.id OR e.target_id = n.id) AS degree
               FROM nodes n
               LEFT JOIN documents d ON d.id = n.first_seen_doc
               ORDER BY n.mention_count DESC"""
        )

        edge_rows = await conn.fetch(
            """SELECT e.id, e.source_id, e.target_id, e.predicate, e.evidence_text,
                      e.confidence, e.document_id, d.title AS doc_title
               FROM edges e
               LEFT JOIN documents d ON d.id = e.document_id"""
        )

    # Group nodes into thematic clusters based on type and connections
    cluster_mapping = {
        "concept": "Foundational Concepts",
        "model": "Model Architectures",
        "method": "Methods & Objectives",
        "dataset": "Benchmarks & Datasets",
        "paper": "Academic Publications",
        "person": "Researchers",
        "other": "General Entities"
    }

    nodes = []
    for r in node_rows:
        n_type = r["node_type"] or "concept"
        nodes.append({
            "id": str(r["id"]),
            "name": r["canonical_name"],
            "canonical_name": r["canonical_name"],
            "label": r["canonical_name"],
            "type": n_type,
            "cluster": cluster_mapping.get(n_type, "General Entities"),
            "description": r["description"] or f"Entity {r['canonical_name']} identified in knowledge corpus.",
            "aliases": r["aliases"] or [],
            "mention_count": r["mention_count"] or 1,
            "degree": r["degree"] or 1,
            "confidence": 0.95,
            "document_id": str(r["first_seen_doc"]) if r["first_seen_doc"] else None,
            "doc_title": r["doc_title"],
            "size": max(7.0, min(24.0, (r["degree"] or 1) * 2.2 + 6.0))
        })

    edges = []
    for r in edge_rows:
        edges.append({
            "id": str(r["id"]),
            "source": str(r["source_id"]),
            "target": str(r["target_id"]),
            "source_id": str(r["source_id"]),
            "target_id": str(r["target_id"]),
            "type": r["predicate"],
            "predicate": r["predicate"],
            "evidence": r["evidence_text"],
            "confidence": r["confidence"] or "high",
            "document_id": str(r["document_id"]) if r["document_id"] else None,
            "doc_title": r["doc_title"]
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "links": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges)
    }


@router.get("/nodes/{node_id}")
async def get_node_detail(node_id: UUID):
    """Retrieve full concept profile, linked citations, and neighboring relationships."""
    async with db.pool().acquire() as conn:
        node = await conn.fetchrow(
            """SELECT n.*, d.title AS doc_title FROM nodes n
               LEFT JOIN documents d ON d.id = n.first_seen_doc
               WHERE n.id = $1""", node_id
        )
        if not node:
            raise HTTPException(404, "Concept node not found")

        edges = await conn.fetch(
            """SELECT e.*, 
                      ns.canonical_name AS source_name, ns.node_type AS source_type,
                      nt.canonical_name AS target_name, nt.node_type AS target_type,
                      d.title AS doc_title
               FROM edges e
               JOIN nodes ns ON ns.id = e.source_id
               JOIN nodes nt ON nt.id = e.target_id
               LEFT JOIN documents d ON d.id = e.document_id
               WHERE e.source_id = $1 OR e.target_id = $1""", node_id
        )

        # Retrieve relevant chunks mentioning this concept
        chunks = await conn.fetch(
            """SELECT c.id, c.document_id, c.chunk_index, c.section_path, c.page_start, c.text, d.title AS doc_title
               FROM chunks c
               JOIN documents d ON d.id = c.document_id
               WHERE c.tsv @@ plainto_tsquery('english', $1)
               LIMIT 5""", node["canonical_name"]
        )

    return {
        "node": {
            "id": str(node["id"]),
            "name": node["canonical_name"],
            "aliases": node["aliases"] or [],
            "type": node["node_type"],
            "description": node["description"],
            "mention_count": node["mention_count"],
            "first_seen_doc": str(node["first_seen_doc"]) if node["first_seen_doc"] else None,
            "doc_title": node["doc_title"],
            "confidence": 0.96
        },
        "relationships": [
            {
                "id": str(e["id"]),
                "source_id": str(e["source_id"]),
                "target_id": str(e["target_id"]),
                "source_name": e["source_name"],
                "target_name": e["target_name"],
                "predicate": e["predicate"],
                "evidence_text": e["evidence_text"],
                "confidence": e["confidence"],
                "doc_title": e["doc_title"]
            }
            for e in edges
        ],
        "evidence_chunks": [
            {
                "id": str(c["id"]),
                "document_id": str(c["document_id"]),
                "chunk_index": c["chunk_index"],
                "section_path": c["section_path"],
                "page": c["page_start"],
                "text": c["text"],
                "doc_title": c["doc_title"]
            }
            for c in chunks
        ]
    }


@router.get("/stats")
async def get_graph_stats():
    """Summary metrics of the knowledge graph."""
    async with db.pool().acquire() as conn:
        node_count = await conn.fetchval("SELECT count(*) FROM nodes")
        edge_count = await conn.fetchval("SELECT count(*) FROM edges")
        doc_count = await conn.fetchval("SELECT count(*) FROM documents WHERE status = 'done'")
        contradictions = await conn.fetchval("SELECT count(*) FROM edges WHERE predicate = 'CONTRADICTS'")

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "document_count": doc_count,
        "contradiction_count": contradictions,
        "status": "active"
    }
