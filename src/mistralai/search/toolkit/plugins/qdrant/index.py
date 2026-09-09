from typing import Any, override

from mistralai.search.toolkit.context import IngestContext, RetrievalContext
from mistralai.search.toolkit.document import (
    ChunkPatch,
    ChunkType,
    Document,
    DocumentPatch,
)
from mistralai.search.toolkit.search import (
    GrepMode,
    NavigableIndex,
    NavigationDirection,
    SearchResult,
    SourceNotFoundError,
    VectorSearchQuery,
)
from mistralai.search.toolkit.search.errors import (
    ChunkNotFoundError,
    DocumentNotFoundError,
    IndexingError,
    SearchError,
)
from mistralai.search.toolkit.search.index import PatchableIndex, VectorStoreIndex
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models
from structlog import get_logger

from mistralai.search.toolkit.plugins.qdrant import mapping
from mistralai.search.toolkit.plugins.qdrant.schema import QdrantCollectionSchema

logger = get_logger(__name__)

_DEPTH = 40


class QdrantSearchQuery(VectorSearchQuery):
    vector_weight: float = Field(default=1.0, ge=0.0)
    text_weight: float = Field(default=1.0, ge=0.0)
    rrf_k: int = Field(default=4, ge=0)


def _eq(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


class QdrantStoreIndex(VectorStoreIndex, NavigableIndex, PatchableIndex):
    def __init__(
        self,
        client: AsyncQdrantClient,
        schema: QdrantCollectionSchema,
        *,
        owns_client: bool = False,
    ) -> None:
        self._client = client
        self._schema = schema
        self._name = schema.collection_name
        self._owns_client = owns_client
        self._metric = schema.embedding_model.distance_metric
        self._custom = schema.custom_mappings()
        self._keys = tuple(m.key for m in self._custom)
        self._patch_keys = {
            src: {m.field: m.key for m in self._custom if m.source == src}
            for src in ("chunk", "document")
        }

    def _dim(self) -> int:
        return self._schema.embedding_model.dimensions  # type: ignore[union-attr]

    def _filter_out(self, exclude_ids: set[str]) -> models.Filter | None:
        if not exclude_ids:
            return None
        return models.Filter(
            must_not=[
                models.HasIdCondition(has_id=[mapping.point_id(i) for i in exclude_ids])
            ]
        )

    async def _scroll(
        self,
        filt: models.Filter,
        *,
        limit: int | None = None,
        order: models.Direction | None = None,
        with_vectors: bool = False,
    ) -> list[models.Record]:
        out: list[models.Record] = []
        offset: Any = None
        left = limit
        while True:
            page, offset = await self._client.scroll(
                collection_name=self._name,
                scroll_filter=filt,
                limit=1000 if left is None else min(left, 1000),
                offset=offset,
                order_by=models.OrderBy(key="start_offset", direction=order)
                if order
                else None,
                with_payload=True,
                with_vectors=with_vectors,
            )
            out.extend(page)
            if left is not None:
                left -= len(page)
                if left <= 0:
                    return out[:limit]
            if offset is None:
                return out

    async def _get(
        self, chunk_id: str, *, with_vectors: bool = False
    ) -> models.Record | None:
        hits = await self._client.retrieve(
            collection_name=self._name,
            ids=[mapping.point_id(chunk_id)],
            with_payload=True,
            with_vectors=with_vectors,
        )
        if hits:
            return hits[0]
        page, _ = await self._client.scroll(
            collection_name=self._name,
            scroll_filter=models.Filter(must=[_eq("id", chunk_id)]),
            limit=1,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return page[0] if page else None

    async def _require_source(self, source_id: str) -> None:
        page, _ = await self._client.scroll(
            collection_name=self._name,
            scroll_filter=models.Filter(must=[_eq("source_id", source_id)]),
            limit=1,
            with_payload=False,
        )
        if not page:
            raise SourceNotFoundError(source_id)

    def _results(self, records: list[models.Record]) -> list[SearchResult]:
        return [mapping.unranked(dict(r.payload or {}), self._keys) for r in records]

    def _patch_custom(
        self, patch: BaseModel, base: type[BaseModel], source: str
    ) -> dict[str, Any]:
        known = self._patch_keys[source]
        extra = (
            type(patch).model_fields.keys() - base.model_fields.keys()
        ) & patch.model_fields_set
        out: dict[str, Any] = {}
        for name in extra:
            key = known.get(name)
            if key is None:
                raise IndexingError(
                    f"Patch field {name!r} is not a custom {source} field (known: {sorted(known)})"
                )
            out[key] = mapping.payload_value(getattr(patch, name))
        return out

    def _merge_meta(
        self, current: dict[str, Any], metadata: BaseModel, prefix: str = ""
    ) -> dict[str, Any]:
        merged = dict(current)
        dumped = metadata.model_dump(mode="json")
        for key in metadata.model_fields_set | set((metadata.model_extra or {}).keys()):
            blob_key = f"{prefix}{key}"
            value = dumped.get(key)
            if value is None:
                merged.pop(blob_key, None)
            else:
                merged[blob_key] = value
        return merged

    @override
    async def index_document(
        self, document: Document, context: IngestContext = IngestContext()
    ) -> None:
        if not document.chunks:
            raise IndexingError(
                "No chunks for document; use delete_document() to remove it"
            )
        if not any(c.embedding for c in document.chunks):
            raise IndexingError("No chunks with embeddings for document")
        dim = self._dim()
        points: list[models.PointStruct] = []
        skipped = 0
        for chunk in document.chunks:
            if not chunk.embedding:
                skipped += 1
                continue
            if len(chunk.embedding) != dim:
                raise IndexingError(
                    f"Embedding dimension mismatch for chunk {chunk.id}: expected {dim}, got {len(chunk.embedding)}"
                )
            points.append(
                models.PointStruct(
                    id=mapping.point_id(chunk.id),
                    vector=list(chunk.embedding),
                    payload=mapping.to_payload(document, chunk, self._custom),
                )
            )
        if skipped:
            logger.warning(
                "Skipping chunks without an embedding. Positional reads will skip those ranges.",
                document_id=document.id,
                skipped=skipped,
                indexed=len(points),
            )
        try:
            await self._client.upsert(
                collection_name=self._name, points=points, wait=True
            )
            await self._client.delete(
                collection_name=self._name,
                points_selector=models.Filter(
                    must=[_eq("document_id", document.id)],
                    must_not=[models.HasIdCondition(has_id=[p.id for p in points])],
                ),
                wait=True,
            )
        except Exception as exc:
            raise IndexingError("Failed to index document") from exc

    @override
    async def delete_document(
        self, doc_id: str, context: IngestContext = IngestContext()
    ) -> None:
        filt = models.Filter(must=[_eq("document_id", doc_id)])
        try:
            n = await self._client.count(
                collection_name=self._name, count_filter=filt, exact=True
            )
            if n.count == 0:
                raise DocumentNotFoundError(doc_id)
            await self._client.delete(
                collection_name=self._name, points_selector=filt, wait=True
            )
        except DocumentNotFoundError:
            raise
        except Exception as exc:
            raise IndexingError(f"Failed to delete document {doc_id}") from exc

    @override
    async def search(
        self, query: VectorSearchQuery, context: RetrievalContext = RetrievalContext()
    ) -> list[SearchResult]:
        if query.top_k < 1:
            raise SearchError("top_k must be at least 1")
        mc = query.approximate_options.max_candidates
        if mc is not None and mc < 1:
            raise SearchError("max_candidates must be at least 1")
        q = (
            query
            if isinstance(query, QdrantSearchQuery)
            else QdrantSearchQuery(**query.model_dump())
        )
        ef = (
            None
            if mc is None and query.top_k <= _DEPTH
            else max(mc or _DEPTH, query.top_k)
        )
        params = models.SearchParams(hnsw_ef=ef) if ef else None
        if q.query and q.query.strip():
            return await self._hybrid(q, params)
        try:
            response = await self._client.query_points(
                collection_name=self._name,
                query=list(query.embedding),
                query_filter=self._filter_out(query.exclude_ids),
                search_params=params,
                limit=query.top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise SearchError("Search query failed", query=query.query) from exc
        return [
            mapping.ranked(
                dict(p.payload or {}),
                p.score,
                self._metric,
                self._keys,
                query.include_content,
                query.include_metadata,
            )
            for p in response.points
        ]

    async def _hybrid(
        self, query: QdrantSearchQuery, params: models.SearchParams | None
    ) -> list[SearchResult]:
        depth = max(query.top_k, query.approximate_options.max_candidates or _DEPTH)
        try:
            dense = await self._client.query_points(
                collection_name=self._name,
                query=list(query.embedding),
                query_filter=self._filter_out(query.exclude_ids),
                search_params=params,
                limit=depth,
                with_payload=True,
            )
            lexical = await self._scroll(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content",
                            match=models.MatchText(text=query.query or ""),
                        )
                    ]
                ),
                limit=depth,
            )
        except Exception as exc:
            raise SearchError("Search query failed", query=query.query) from exc
        dense_ids = [str((p.payload or {}).get("id", p.id)) for p in dense.points]
        payloads = {
            cid: dict(p.payload or {}) for cid, p in zip(dense_ids, dense.points)
        }
        lexical_ids: list[str] = []
        for rec in lexical:
            payload = dict(rec.payload or {})
            cid = str(payload.get("id", rec.id))
            if cid not in query.exclude_ids:
                payloads.setdefault(cid, payload)
                lexical_ids.append(cid)
        return mapping.fuse(
            [(cid, p.score) for cid, p in zip(dense_ids, dense.points)],
            lexical_ids,
            payloads,
            self._metric,
            self._keys,
            top_k=query.top_k,
            vector_weight=query.vector_weight,
            text_weight=query.text_weight,
            rrf_k=query.rrf_k,
            include_content=query.include_content,
            include_metadata=query.include_metadata,
        )

    async def _span(
        self,
        source_id: str,
        must: list[models.Condition],
        *,
        top_k: int,
        reverse: bool = False,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise SearchError("top_k must be at least 1")
        try:
            records = await self._scroll(
                models.Filter(must=must),
                limit=top_k,
                order=models.Direction.DESC if reverse else models.Direction.ASC,
            )
        except Exception as exc:
            raise SearchError("Positional query failed") from exc
        if not records:
            await self._require_source(source_id)
            return []
        if reverse:
            records.reverse()
        return self._results(records)

    @override
    async def navigate(
        self,
        source_id: str,
        start_offset: int,
        end_offset: int,
        direction: NavigationDirection,
        *,
        top_k: int = 1,
        content_type: ChunkType = ChunkType.CONTENT,
        context: RetrievalContext = RetrievalContext(),
    ) -> list[SearchResult]:
        nxt = direction == NavigationDirection.NEXT
        bound = (
            models.FieldCondition(
                key="start_offset", range=models.Range(gte=end_offset)
            )
            if nxt
            else models.FieldCondition(
                key="end_offset", range=models.Range(lte=start_offset)
            )
        )
        return await self._span(
            source_id,
            [_eq("source_id", source_id), _eq("chunk_type", content_type.value), bound],
            top_k=top_k,
            reverse=not nxt,
        )

    @override
    async def read(
        self,
        source_id: str,
        start_offset: int | None,
        end_offset: int | None,
        *,
        content_type: ChunkType = ChunkType.CONTENT,
        top_k: int = 20,
        context: RetrievalContext = RetrievalContext(),
    ) -> list[SearchResult]:
        must: list[models.Condition] = [
            _eq("source_id", source_id),
            _eq("chunk_type", content_type.value),
        ]
        if start_offset is not None:
            must.append(
                models.FieldCondition(
                    key="start_offset", range=models.Range(gte=start_offset)
                )
            )
        if end_offset is not None:
            must.append(
                models.FieldCondition(
                    key="end_offset", range=models.Range(lte=end_offset)
                )
            )
        return await self._span(source_id, must, top_k=top_k)

    @override
    async def grep(
        self,
        source_id: str,
        pattern: str,
        *,
        mode: GrepMode = GrepMode.PHRASE,
        content_type: ChunkType = ChunkType.CONTENT,
        top_k: int = 5,
        context: RetrievalContext = RetrievalContext(),
    ) -> list[SearchResult]:
        if not pattern.strip():
            await self._require_source(source_id)
            return []
        match = (
            models.MatchPhrase(phrase=pattern)
            if mode == GrepMode.PHRASE
            else models.MatchText(text=pattern)
        )
        return await self._span(
            source_id,
            [
                _eq("source_id", source_id),
                _eq("chunk_type", content_type.value),
                models.FieldCondition(key="content", match=match),
            ],
            top_k=top_k,
        )

    @override
    async def get_chunk(
        self, chunk_id: str, *, context: RetrievalContext = RetrievalContext()
    ) -> SearchResult | None:
        try:
            rec = await self._get(chunk_id)
        except Exception as exc:
            raise SearchError(f"Failed to fetch chunk {chunk_id!r}") from exc
        return mapping.unranked(dict(rec.payload or {}), self._keys) if rec else None

    @override
    async def patch_chunk(
        self, chunk_id: str, patch: ChunkPatch, context: IngestContext = IngestContext()
    ) -> None:
        try:
            rec = await self._get(chunk_id, with_vectors=True)
        except Exception as exc:
            raise IndexingError("Failed to apply patch") from exc
        if rec is None:
            raise ChunkNotFoundError(chunk_id)
        payload = dict(rec.payload or {})
        vector = list(rec.vector) if rec.vector is not None else None
        if patch.content is not None:
            payload["content"] = patch.content
        if patch.embedding is not None:
            dim = self._dim()
            if len(patch.embedding) != dim:
                raise IndexingError(
                    f"Embedding dimension mismatch: expected {dim}, got {len(patch.embedding)}"
                )
            vector = list(patch.embedding)
        if patch.metadata is not None:
            payload["metadata"] = self._merge_meta(
                dict(payload.get("metadata") or {}), patch.metadata
            )
        payload.update(self._patch_custom(patch, ChunkPatch, "chunk"))
        try:
            await self._client.upsert(
                collection_name=self._name,
                points=[models.PointStruct(id=rec.id, vector=vector, payload=payload)],
                wait=True,
            )
        except Exception as exc:
            raise IndexingError("Failed to apply patch") from exc

    @override
    async def patch_document(
        self,
        document_id: str,
        patch: DocumentPatch,
        context: IngestContext = IngestContext(),
    ) -> None:
        try:
            records = await self._scroll(
                models.Filter(must=[_eq("document_id", document_id)]), with_vectors=True
            )
        except Exception as exc:
            raise IndexingError("Failed to apply patch") from exc
        if not records:
            raise DocumentNotFoundError(document_id)
        extra = self._patch_custom(patch, DocumentPatch, "document")
        points = []
        for rec in records:
            payload = dict(rec.payload or {})
            if patch.metadata is not None:
                payload["metadata"] = self._merge_meta(
                    dict(payload.get("metadata") or {}),
                    patch.metadata,
                    prefix="document_",
                )
            payload.update(extra)
            points.append(
                models.PointStruct(
                    id=rec.id,
                    vector=list(rec.vector) if rec.vector is not None else None,
                    payload=payload,
                )
            )
        try:
            await self._client.upsert(
                collection_name=self._name, points=points, wait=True
            )
        except Exception as exc:
            raise IndexingError("Failed to apply patch") from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()
