"""Qdrant wrapper.

Why Qdrant:
  - Rust-native, single binary, easy to deploy on K8s.
  - HNSW index with tunable `m` (connections per node) and `ef_construct` (search width).
  - Payload filtering for hybrid search (e.g., "only diffusion defects from line A").
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import settings


@dataclass
class RetrievedCase:
    """A case fetched from the vector DB, with similarity score and metadata."""
    score: float
    payload: dict[str, Any]


class VectorStore:
    """Thin wrapper around QdrantClient with the patterns we actually need."""

    def __init__(self, collection: str | None = None) -> None:
        self.collection = collection or settings.qdrant_collection
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

    def reset_collection(self) -> None:
        """Drop and recreate the collection. Useful for re-ingestion."""
        collections = {c.name for c in self.client.get_collections().collections}
        if self.collection in collections:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    def upsert(self, vector: list[float], payload: dict[str, Any]) -> str:
        """Insert one point. Returns the generated UUID."""
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return point_id

    def upsert_batch(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> list[str]:
        """Batch upsert — much faster than one-at-a-time for ingestion."""
        assert len(vectors) == len(payloads), "vectors and payloads must align"
        ids = [str(uuid.uuid4()) for _ in vectors]
        points = [
            PointStruct(id=pid, vector=vec, payload=pl)
            for pid, vec, pl in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return ids

    def search(
        self,
        query_vector: list[float],
        top_k: int = 2,
        filter_: dict | None = None,
    ) -> list[RetrievedCase]:
        """Cosine similarity search. Optionally filter by payload."""
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=filter_,
        )
        return [RetrievedCase(score=h.score, payload=h.payload or {}) for h in result.points]
