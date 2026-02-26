"""
embeddings.py - Vector embedding layer using Qdrant and BAAI/bge-m3.

Provides:
- EmbeddingManager: wraps sentence-transformers + Qdrant
- build_summary(node) -> str: text used for embedding
- build_embeddings(nodes): bulk initial indexing
- update_embeddings(added, updated, deleted): incremental update
- search(query, limit) -> list[SearchResult]: semantic search

Two encoding modes:
  LOCAL:  load model on this machine (device = "cpu" or "cuda")
  REMOTE: delegate encoding to embedding_server.py running on a GPU machine
          → set EmbeddingConfig(remote_url="http://gpu-host:8765")

Requires qdrant-client>=1.9 and a running Qdrant server (see docker/run-qdrant.sh).
Supports local file mode via QdrantConfig(local_path="./data").
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import numpy as np
from neo4j import Driver
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PointIdsList,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

from ast_rag.models import ASTNode, SearchResult, QdrantConfig, EmbeddingConfig, NodeKind

logger = logging.getLogger(__name__)

# Node kinds that are worth embedding (ignore field-level noise)
EMBEDDABLE_KINDS = frozenset([
    NodeKind.CLASS, NodeKind.INTERFACE, NodeKind.STRUCT, NodeKind.ENUM,
    NodeKind.TRAIT, NodeKind.FUNCTION, NodeKind.METHOD,
    NodeKind.CONSTRUCTOR, NodeKind.DESTRUCTOR,
])


def build_summary(node: ASTNode) -> str:
    """Build a short natural-language summary of an AST node for embedding.

    Format: '<lang> <kind>: <qualified_name> | signature: <sig> | file: <path>:<line>'
    """
    sig_part = f" | signature: {node.signature}" if node.signature else ""
    return (
        f"{node.lang.value} {node.kind.value}: {node.qualified_name}"
        f"{sig_part}"
        f" | file: {node.file_path}:{node.start_line}"
    )


def _node_id_to_point_id(node_id: str) -> str:
    """Convert a 24-char hex node ID to a deterministic UUID string for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ast-rag:{node_id}"))


class EmbeddingManager:
    """Manages Qdrant collection + bge-m3 sentence-transformer model.

    Usage::

        em = EmbeddingManager(qdrant_cfg, embed_cfg)
        em.build_embeddings(nodes)
        results = em.search("function that processes HTTP requests", limit=10)
    """

    def __init__(
        self,
        qdrant_config: QdrantConfig,
        embed_config: EmbeddingConfig,
        neo4j_driver: Optional[Driver] = None,
    ) -> None:
        self._qdrant_config = qdrant_config
        self._embed_config = embed_config
        self._neo4j_driver = neo4j_driver
        self._client: Optional[QdrantClient] = None
        self._model: Optional[SentenceTransformer] = None
        # Validate hybrid weights if hybrid search is enabled
        if self._embed_config.hybrid_search:
            total = self._embed_config.vector_weight + self._embed_config.keyword_weight
            if abs(total - 1.0) > 0.01:
                logger.warning(
                    "Hybrid weights sum to %.2f (expected ~1.0). "
                    "Consider adjusting vector_weight and keyword_weight.",
                    total
                )

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        """Return local SentenceTransformer.  Not called when remote_url is set."""
        if self._model is None:
            logger.info("Loading embedding model locally: %s (device=%s)",
                        self._embed_config.model_name, self._embed_config.device)
            self._model = SentenceTransformer(
                self._embed_config.model_name,
                device=self._embed_config.device,
            )
        return self._model

    def _encode(self, texts: list[str]) -> "np.ndarray":
        """Encode a list of texts into L2-normalised embedding vectors.

        Routes to remote HTTP API when EmbeddingConfig.remote_url is set;
        otherwise runs the model locally.

        Returns:
            numpy float32 array of shape (len(texts), dim)
        """
        if self._embed_config.remote_url:
            return self._encode_remote(texts)
        return self._encode_local(texts)

    def _encode_local(self, texts: list[str]) -> "np.ndarray":
        model = self._get_model()
        return model.encode(texts, normalize_embeddings=True)  # type: ignore[return-value]

    def _encode_remote(self, texts: list[str]) -> "np.ndarray":
        """Call a remote embedding API to encode texts.

        Supports two protocols based on `remote_url`:

        1. **Our embedding_server.py** (default):
           POST {remote_url}/embed
           Body:     {"texts": [...], "normalize": true}
           Response: {"embeddings": [[...], ...], "dim": 1024}

        2. **OpenAI-compatible API** (Ollama, HuggingFace TEI, LiteLLM, etc.):
           Detected when remote_url ends with "/v1" or contains "/v1/embeddings".
           POST {remote_url}/embeddings
           Body:     {"model": "<model_name>", "input": [...]}
           Response: {"data": [{"embedding": [...], ...}, ...]}

        Set remote_url examples:
          - Our server:   "http://gpu-host:8765"
          - Ollama:       "http://gpu-host:11434/v1"
          - TEI:          "http://tei-host:8080/v1"

        Sub-batching: if len(texts) > remote_batch_size, the request is split
        into sub-batches and results are concatenated.

        Raises RuntimeError on HTTP error.
        """
        batch_size = self._embed_config.remote_batch_size
        if len(texts) > batch_size:
            logger.debug(
                "Sub-batching %d texts into chunks of %d for remote encoding",
                len(texts), batch_size,
            )
            parts: list[np.ndarray] = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                parts.append(self._encode_remote_batch(chunk))
            return np.concatenate(parts, axis=0)

        return self._encode_remote_batch(texts)

    def _encode_remote_batch(self, texts: list[str]) -> "np.ndarray":
        """Send a single batch of texts to the remote embedding server.

        This is the low-level method called by _encode_remote() after
        sub-batching has been applied.
        """
        import httpx  # lazy import — not needed for local mode

        base = self._embed_config.remote_url.rstrip("/")

        # Detect OpenAI-compatible endpoint
        if "/v1" in base or base.endswith("/embeddings"):
            url = base if base.endswith("/embeddings") else base + "/embeddings"
            payload = {
                "model": self._embed_config.model_name,
                "input": texts,
            }
            resp_key = "data"
        else:
            # Our custom embedding_server.py protocol
            url = base + "/embed"
            payload = {"texts": texts, "normalize": True}
            resp_key = None  # top-level "embeddings" key

        try:
            resp = httpx.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body_preview = exc.response.text[:500] if exc.response else "(no response body)"
            logger.error(
                "Remote embedding server returned HTTP %s: %s",
                exc.response.status_code if exc.response else "???",
                body_preview,
            )
            raise RuntimeError(
                f"Remote embedding server HTTP {exc.response.status_code}: {body_preview}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Remote embedding server error: {exc}") from exc

        data = resp.json()

        if resp_key == "data":
            # OpenAI format: {"data": [{"embedding": [...], ...}, ...]}
            raw = [item["embedding"] for item in data["data"]]
        else:
            # Our format: {"embeddings": [[...], ...]}
            raw = data["embeddings"]

        embeddings = np.array(raw, dtype=np.float32)
        logger.debug("Remote encoded %d texts → shape %s via %s", len(texts), embeddings.shape, url)
        return embeddings

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            cfg = self._qdrant_config
            if cfg.local_path:
                self._client = QdrantClient(path=cfg.local_path)
                logger.info("Qdrant local mode at %s", cfg.local_path)
            else:
                self._client = QdrantClient(url=cfg.url)
                logger.info("Qdrant remote at %s", cfg.url)
            self._ensure_collection(self._client)
        return self._client

    def _ensure_collection(self, client: QdrantClient) -> None:
        """Create collection if it doesn't already exist."""
        name = self._qdrant_config.collection_name
        if not client.collection_exists(name):
            model = self._get_model()
            dim = model.get_sentence_embedding_dimension()
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", name, dim)
        else:
            logger.info("Using existing Qdrant collection '%s'", name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_embeddings(
        self,
        nodes: list[ASTNode],
        batch_size: int = 64,
    ) -> int:
        """Build embeddings for all embeddable nodes (bulk).

        Returns the number of nodes indexed.
        """
        to_embed = [n for n in nodes if n.kind in EMBEDDABLE_KINDS]
        if not to_embed:
            return 0

        logger.info("Building embeddings for %d nodes...", len(to_embed))
        client = self._get_client()
        name = self._qdrant_config.collection_name

        count = 0
        for i in range(0, len(to_embed), batch_size):
            batch = to_embed[i : i + batch_size]
            texts = [build_summary(n) for n in batch]
            embeddings = self._encode(texts)

            points = [
                PointStruct(
                    id=_node_id_to_point_id(n.id),
                    vector=embeddings[j].tolist(),
                    payload=_node_to_payload(n),
                )
                for j, n in enumerate(batch)
            ]
            client.upsert(collection_name=name, points=points, wait=True)
            count += len(batch)
            logger.debug(
                "Embedded batch %d/%d",
                i // batch_size + 1,
                (len(to_embed) + batch_size - 1) // batch_size,
            )

        logger.info("Embedding complete: %d nodes indexed.", count)
        return count

    def update_embeddings(
        self,
        added: list[ASTNode],
        updated: list[ASTNode],
        deleted_ids: list[str],
    ) -> None:
        """Incremental embedding update.

        - Delete embeddings for deleted_ids.
        - Upsert embeddings for added + updated nodes.
        """
        client = self._get_client()
        name = self._qdrant_config.collection_name

        # Delete old embeddings
        if deleted_ids:
            point_ids = [_node_id_to_point_id(nid) for nid in deleted_ids]
            try:
                client.delete(
                    collection_name=name,
                    points_selector=PointIdsList(points=point_ids),
                    wait=True,
                )
                logger.debug("Deleted %d embeddings", len(deleted_ids))
            except Exception as exc:
                logger.warning("Failed to delete embeddings: %s", exc)

        # Upsert new / updated embeddings
        to_upsert = [n for n in (added + updated) if n.kind in EMBEDDABLE_KINDS]
        if to_upsert:
            self.build_embeddings(to_upsert)

    def search(
        self,
        query: str,
        limit: int = 10,
        lang_filter: Optional[str] = None,
        kind_filter: Optional[str] = None,
        auto_fallback: bool = False,
    ) -> list[SearchResult]:
        """Semantic search over the vector index.

        Args:
            query:       Natural language or code fragment query.
            limit:       Maximum number of results.
            lang_filter: If set, restrict to a specific language.
            kind_filter: If set, restrict to a specific node kind.
            auto_fallback: If True and results < limit, automatically search
                           across all languages to fill remaining slots.

        Returns:
            List of SearchResult sorted by descending similarity.
        """
        client = self._get_client()
        name = self._qdrant_config.collection_name

        vec = self._encode([query])[0].tolist()

        must_conditions = []
        if lang_filter:
            must_conditions.append(
                FieldCondition(key="lang", match=MatchValue(value=lang_filter))
            )
        if kind_filter:
            must_conditions.append(
                FieldCondition(key="kind", match=MatchValue(value=kind_filter))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        try:
            response = client.query_points(
                collection_name=name,
                query=vec,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            logger.error("Qdrant query failed: %s", exc)
            return []

        results: list[SearchResult] = []
        seen_ids: set[str] = set()
        for hit in response.points:
            node = _payload_to_node(hit.payload or {})
            results.append(SearchResult(node=node, score=float(hit.score)))
            seen_ids.add(node.id)

        # Auto-fallback: if results < limit and filters were applied, search without filters
        if auto_fallback and len(results) < limit and (lang_filter or kind_filter):
            remaining = limit - len(results)
            logger.debug(
                "Only %d results with filters (lang=%s, kind=%s), "
                "searching across all languages for %d more",
                len(results), lang_filter, kind_filter, remaining
            )
            
            # Search without language filter to fill remaining
            fallback_response = client.query_points(
                collection_name=name,
                query=vec,
                query_filter=None,  # No filters
                limit=remaining,
                with_payload=True,
            )
            
            for hit in fallback_response.points:
                node = _payload_to_node(hit.payload or {})
                # Avoid duplicates
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    results.append(SearchResult(node=node, score=float(hit.score)))

        return results

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        client = self._get_client()
        name = self._qdrant_config.collection_name
        return client.count(collection_name=name).count

    # ------------------------------------------------------------------
    # Hybrid Search: Vector + Keyword fusion
    # ------------------------------------------------------------------

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Min-max normalize scores to [0, 1].

        Handles edge cases:
        - Empty list: returns empty list
        - All equal scores: returns all 1.0
        """
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0 for _ in scores]  # All equal, treat as max
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _keyword_search(
        self,
        query: str,
        limit: int,
        lang_filter: Optional[str] = None,
    ) -> list[tuple[ASTNode, float]]:
        """Full-text keyword search via Neo4j full-text index.

        Uses the ast_symbol_fulltext index created in schema/graph_schema.cql.

        Args:
            query: Search query (keywords)
            limit: Maximum number of results
            lang_filter: Optional language filter

        Returns:
            List of (ASTNode, score) tuples, where score is from Neo4j (0.0-1.0)
        """
        if self._neo4j_driver is None:
            logger.warning("Neo4j driver not configured for keyword search")
            return []

        results: list[tuple[ASTNode, float]] = []

        # Build Cypher query for full-text search
        lang_clause = "AND node.lang = $lang" if lang_filter else ""

        cypher = """
CALL db.index.fulltext.queryNodes(
    'ast_symbol_fulltext',
    $search_query,
    {limit: $limit}
)
YIELD node, score
WHERE node.valid_to IS NULL
  """ + lang_clause + """
RETURN node, score
ORDER BY score DESC
"""
        params: dict = {
            "search_query": query,
            "limit": limit,
        }
        if lang_filter:
            params["lang"] = lang_filter

        try:
            with self._neo4j_driver.session() as session:
                for record in session.run(cypher, **params):
                    node_data = dict(record["node"])
                    score = float(record["score"])
                    node = ASTNode(
                        id=node_data.get("id", ""),
                        kind=NodeKind(node_data.get("kind", "Function")),
                        name=node_data.get("name", ""),
                        qualified_name=node_data.get("qualified_name", ""),
                        lang=node_data.get("lang", "java"),
                        file_path=node_data.get("file_path", ""),
                        start_line=int(node_data.get("start_line", 0)),
                        end_line=int(node_data.get("end_line", 0)),
                        start_byte=int(node_data.get("start_byte", 0)),
                        end_byte=int(node_data.get("end_byte", 0)),
                        signature=node_data.get("signature") or None,
                        valid_from=node_data.get("valid_from", "INIT"),
                        valid_to=node_data.get("valid_to"),
                    )
                    results.append((node, score))
        except Exception as exc:
            logger.warning("Full-text search failed: %s", exc)

        return results

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        lang_filter: Optional[str] = None,
        kind_filter: Optional[str] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
    ) -> list[SearchResult]:
        """Combine vector and keyword search with weighted linear fusion.

        Fusion formula:
            final_score = w_vec * normalized_vector_score + w_kw * normalized_keyword_score

        Args:
            query: Natural language or code query
            limit: Maximum results to return
            lang_filter: Optional language filter (applies to both searches)
            kind_filter: Optional node kind filter (vector search only)
            vector_weight: Override config vector weight (None = use config)
            keyword_weight: Override config keyword weight (None = use config)

        Returns:
            List of SearchResult ordered by fused score (descending)
        """
        # Use config weights if not overridden
        if vector_weight is None:
            vector_weight = self._embed_config.vector_weight
        if keyword_weight is None:
            keyword_weight = self._embed_config.keyword_weight

        # Normalize weights to sum to 1.0
        total = vector_weight + keyword_weight
        if total > 0:
            vector_weight = vector_weight / total
            keyword_weight = keyword_weight / total
        # Get results from both sources
        vector_results = self.search(
            query=query,
            limit=limit * 2,  # Get more for fusion
            lang_filter=lang_filter,
            kind_filter=kind_filter,
            auto_fallback=False,  # We handle fallback ourselves
        )  # List[SearchResult]

        keyword_results = self._keyword_search(
            query=query,
            limit=limit * 2,
            lang_filter=lang_filter,
        )  # List[(ASTNode, float)]

        # Extract scores for normalization
        vector_scores = [r.score for r in vector_results]
        keyword_scores = [s for _, s in keyword_results]

        # Normalize scores to [0, 1]
        normalized_vector_scores = self._normalize_scores(vector_scores)
        normalized_keyword_scores = self._normalize_scores(keyword_scores)

        # Build lookup dicts: node_id -> (node, normalized_score)
        vector_dict: dict[str, tuple[ASTNode, float]] = {}
        for i, r in enumerate(vector_results):
            vector_dict[r.node.id] = (r.node, normalized_vector_scores[i] if i < len(normalized_vector_scores) else 0.0)

        keyword_dict: dict[str, tuple[ASTNode, float]] = {}
        for i, (node, score) in enumerate(keyword_results):
            keyword_dict[node.id] = (node, normalized_keyword_scores[i] if i < len(normalized_keyword_scores) else 0.0)

        # Combine: union of both result sets
        all_node_ids = set(vector_dict.keys()) | set(keyword_dict.keys())
        fused_results: list[SearchResult] = []

        for node_id in all_node_ids:
            vec_node, vec_score = vector_dict.get(node_id, (None, 0.0))
            kw_node, kw_score = keyword_dict.get(node_id, (None, 0.0))

            # Use whichever node we have
            node = vec_node or kw_node

            # Compute fused score
            final_score = vector_weight * vec_score + keyword_weight * kw_score

            if node:
                fused_results.append(SearchResult(node=node, score=final_score))

        # Sort by final score descending and return top-k
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:limit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_to_payload(node: ASTNode) -> dict:
    """Convert an ASTNode to a Qdrant payload dict."""
    return {
        "node_id": node.id,
        "name": node.name,
        "qualified_name": node.qualified_name,
        "kind": node.kind.value,
        "lang": node.lang.value,
        "file_path": node.file_path,
        "start_line": node.start_line,
        "end_line": node.end_line,
        "signature": node.signature or "",
        "valid_from": node.valid_from,
    }


def _payload_to_node(payload: dict) -> ASTNode:
    """Reconstruct a minimal ASTNode from Qdrant payload (for search results)."""
    from ast_rag.models import Language as Lang
    return ASTNode(
        id=payload.get("node_id", ""),
        kind=NodeKind(payload.get("kind", "Function")),
        name=payload.get("name", ""),
        qualified_name=payload.get("qualified_name", ""),
        lang=Lang(payload.get("lang", "java")),
        file_path=payload.get("file_path", ""),
        start_line=int(payload.get("start_line", 0)),
        end_line=int(payload.get("end_line", 0)),
        start_byte=0,
        end_byte=0,
        signature=payload.get("signature") or None,
        valid_from=payload.get("valid_from", "INIT"),
    )
