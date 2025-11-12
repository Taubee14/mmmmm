"""
RAG store utilities for turning historical agent events into retrievable context.

The store tries to use Chroma with persistent disk storage per user/chat.
If Chroma is unavailable it will fall back to an in-memory FAISS (or numpy)
index so that the report generator can still retrieve contextual snippets.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from agentbricks.utils.logger_util import logger

try:  # Optional dependency
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:  # pragma: no cover - exercised in envs without chroma
    chromadb = None
    ChromaSettings = None  # type: ignore

try:  # Optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - exercised on unsupported platforms
    faiss = None

__all__ = ["RagStore", "RagStoreError", "REDACTION_PATTERNS"]

REDACTION_PATTERNS = [
    re.compile(r"API_KEY", flags=re.IGNORECASE),
    re.compile(r"Authorization:\s*[^\s]+", flags=re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9\._\-]+", flags=re.IGNORECASE),
    re.compile(r"(?:/home/[^\s]+|[A-Za-z]:\\[^\s]+)"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{12}\b"
    ),
]

DEFAULT_SENTENCE_MODEL = os.getenv(
    "RAG_FALLBACK_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


class RagStoreError(RuntimeError):
    """Raised when indexing or retrieval fails irrecoverably."""


@dataclass
class _Evidence:
    sequence_number: Optional[int]
    timestamp: str
    text: str
    metadata: Dict[str, Any]
    score: float


class RagStore:
    """Vector-store backed cache for agent events."""

    RETRY_LIMIT = 3

    def __init__(
        self,
        base_dir: str | Path = "chroma-db",
        collection_name: str = "report_events",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.collection_name = collection_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = None
        self._collection = None
        self._chroma_client = None
        self._current_store_key: Optional[str] = None
        self._faiss_index = None
        self._faiss_vectors: Optional[np.ndarray] = None
        self._faiss_records: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    async def index_events(
        self,
        user_id: str,
        chat_id: str,
        events: Sequence[Dict[str, Any]],
    ) -> None:
        """Insert (or upsert) events into the vector store."""
        if not events:
            return

        store_key = self._build_store_key(user_id, chat_id)
        async with self._lock:
            self._ensure_store(store_key)
            prepared = self._prepare_documents(events)
            if not prepared:
                return
            ids, documents, metadatas = prepared
            embeddings = await self._embed_batch(documents)
            normalized = [self._normalize(vec) for vec in embeddings]
            await self._with_retries(
                lambda: self._upsert(ids, normalized, documents, metadatas),
                operation="vector upsert",
            )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the most relevant snippets for the given query."""
        if not query or not query.strip():
            return []

        async with self._lock:
            if not self._collection and not self._faiss_records:
                return []

            sanitized_query = self._redact_text(query.strip())
            embeddings = await self._embed_batch([sanitized_query])
            vector = self._normalize(embeddings[0])
            evidence = await self._with_retries(
                lambda: self._query_index(vector, top_k),
                operation="vector query",
            )
            return [
                {
                    "sequence_number": item.sequence_number,
                    "timestamp": item.timestamp,
                    "text": item.text,
                    "score": item.score,
                }
                for item in evidence
            ]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _ensure_store(self, store_key: str) -> None:
        if store_key == self._current_store_key:
            return
        self._current_store_key = store_key
        if chromadb:
            try:
                persist_path = self.base_dir / store_key
                persist_path.mkdir(parents=True, exist_ok=True)
                if hasattr(chromadb, "PersistentClient"):
                    self._chroma_client = chromadb.PersistentClient(
                        path=str(persist_path)
                    )
                else:  # pragma: no cover - legacy fallback
                    settings = ChromaSettings(
                        persist_directory=str(persist_path),
                        anonymized_telemetry=False,
                    )
                    self._chroma_client = chromadb.Client(settings)
                self._collection = self._chroma_client.get_or_create_collection(
                    name=self.collection_name,
                )
                logger.debug(
                    "Initialized Chroma store at %s for %s",
                    persist_path,
                    store_key,
                )
                # Clear FAISS fallback if we switched back to Chroma
                self._reset_faiss()
                return
            except Exception as exc:  # pragma: no cover - depends on env
                logger.warning(
                    "Failed to init Chroma store (%s), falling back to FAISS: %s",
                    store_key,
                    exc,
                )
                self._collection = None

        # Fallback to in-memory FAISS
        self._reset_faiss()

    def _reset_faiss(self) -> None:
        self._faiss_index = None
        self._faiss_vectors = None
        self._faiss_records = []

    def _build_store_key(self, user_id: str, chat_id: str) -> str:
        safe_user = re.sub(r"[^\w.-]", "_", (user_id or "unknown"))
        safe_chat = re.sub(r"[^\w.-]", "_", (chat_id or "session"))
        return f"{safe_user}_{safe_chat}"

    def _prepare_documents(
        self,
        events: Sequence[Dict[str, Any]],
    ) -> Optional[Tuple[List[str], List[str], List[Dict[str, Any]]]]:
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, event in enumerate(events):
            payload = event.get("payload") or {}
            stage = payload.get("stage")
            text = payload.get("text")
            message = payload.get("message")
            seq = event.get("sequence_number")
            timestamp = self._format_timestamp(event.get("timestamp"))

            parts = [
                f"stage={stage}" if stage else "",
                f"type={payload.get('type')}" if payload.get("type") else "",
                f"text={text}" if text else "",
                f"message={message}" if message else "",
            ]
            raw_text = " | ".join(filter(None, parts))
            if not raw_text:
                continue
            document = self._redact_text(raw_text)
            if not document.strip():
                continue

            metadata = {
                "sequence_number": seq,
                "timestamp": timestamp,
                "stage": stage,
            }
            document_id = f"{seq or idx}-{event.get('timestamp') or 0}"
            ids.append(str(document_id))
            documents.append(document)
            metadatas.append(metadata)

        if not documents:
            return None
        return ids, documents, metadatas

    def _redact_text(self, text: str) -> str:
        sanitized = text
        for pattern in REDACTION_PATTERNS:
            sanitized = pattern.sub("<REDACTED>", sanitized)
        return sanitized

    def _format_timestamp(self, ts: Optional[float]) -> str:
        if not ts:
            return "-"
        try:
            return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:  # pragma: no cover - defensive
            return "-"

    async def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        embedder = self._get_embedder()

        def _execute() -> List[List[float]]:
            return embedder.embed(texts)

        return await self._with_retries(_execute, operation="embedding")

    def _get_embedder(self):
        if self._embedder:
            return self._embedder
        if os.getenv("OPENAI_API_KEY"):
            self._embedder = _OpenAIEmbedder()
        else:
            self._embedder = _SentenceTransformerEmbedder(DEFAULT_SENTENCE_MODEL)
        return self._embedder

    def _normalize(self, vector: Sequence[float]) -> List[float]:
        arr = np.array(vector, dtype="float32")
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr.tolist()
        return (arr / norm).tolist()

    def _upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        if self._collection:
            self._collection.upsert(
                ids=list(ids),
                embeddings=[list(vec) for vec in embeddings],
                documents=list(documents),
                metadatas=list(metadatas),
            )
            return
        self._add_to_faiss(ids, embeddings, documents, metadatas)

    def _add_to_faiss(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        if not embeddings:
            return
        vectors = np.array(embeddings, dtype="float32")
        if self._faiss_vectors is None:
            self._faiss_vectors = vectors
        else:
            self._faiss_vectors = np.vstack([self._faiss_vectors, vectors])

        if faiss:
            if self._faiss_index is None:
                self._faiss_index = faiss.IndexFlatIP(vectors.shape[1])
            self._faiss_index.add(vectors)

        for idx, vector_id in enumerate(ids):
            self._faiss_records.append(
                {
                    "id": vector_id,
                    "document": documents[idx],
                    "metadata": metadatas[idx],
                }
            )

    def _query_index(
        self,
        vector: Sequence[float],
        top_k: int,
    ) -> List[_Evidence]:
        top_k = max(1, top_k)
        if self._collection:
            query = self._collection.query(
                query_embeddings=[list(vector)],
                n_results=top_k,
            )
            return self._build_evidence_from_chroma(query)
        return self._build_evidence_from_faiss(vector, top_k)

    def _build_evidence_from_chroma(self, query: Dict[str, Any]) -> List[_Evidence]:
        metadatas = query.get("metadatas") or [[]]
        documents = query.get("documents") or [[]]
        distances = query.get("distances") or [[]]
        results: List[_Evidence] = []
        for idx in range(len(documents[0])):
            metadata = metadatas[0][idx] if idx < len(metadatas[0]) else {}
            doc = documents[0][idx] if idx < len(documents[0]) else ""
            distance = distances[0][idx] if idx < len(distances[0]) else 0.0
            results.append(
                _Evidence(
                    sequence_number=metadata.get("sequence_number"),
                    timestamp=metadata.get("timestamp", "-"),
                    text=doc,
                    metadata=metadata,
                    score=float(distance),
                )
            )
        return results

    def _build_evidence_from_faiss(
        self,
        vector: Sequence[float],
        top_k: int,
    ) -> List[_Evidence]:
        if self._faiss_vectors is None or not self._faiss_records:
            return []
        query = np.array(vector, dtype="float32").reshape(1, -1)
        if faiss and self._faiss_index is not None:
            distances, indices = self._faiss_index.search(query, top_k)
            indices_row = indices[0]
            sims = distances[0]
        else:
            sims = (self._faiss_vectors @ query.T).reshape(-1)
            indices_row = np.argsort(-sims)[:top_k]

        results: List[_Evidence] = []
        for rank, idx in enumerate(indices_row):
            if idx == -1 or idx >= len(self._faiss_records):
                continue
            record = self._faiss_records[idx]
            metadata = record.get("metadata", {})
            results.append(
                _Evidence(
                    sequence_number=metadata.get("sequence_number"),
                    timestamp=metadata.get("timestamp", "-"),
                    text=record.get("document", ""),
                    metadata=metadata,
                    score=float(sims[rank]) if len(sims) > rank else 0.0,
                )
            )
        return results

    async def _with_retries(self, func, *, operation: str):
        last_error: Optional[Exception] = None
        for attempt in range(1, self.RETRY_LIMIT + 1):
            try:
                return await asyncio.to_thread(func)
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                logger.warning(
                    "RagStore %s failed (attempt %s/%s): %s",
                    operation,
                    attempt,
                    self.RETRY_LIMIT,
                    exc,
                )
        raise RagStoreError(f"{operation} failed after retries") from last_error


class _SentenceTransformerEmbedder:
    """Lazily load the sentence-transformer model."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings.tolist()


class _OpenAIEmbedder:
    """Wrapper around OpenAI embeddings API."""

    def __init__(self) -> None:
        from openai import OpenAI  # type: ignore

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RagStoreError("OPENAI_API_KEY required for OpenAI embeddings")
        base_url = os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]
