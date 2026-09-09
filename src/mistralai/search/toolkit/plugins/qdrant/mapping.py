import uuid
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from typing import Any

from mistralai.search.toolkit.document import Document, DocumentChunk
from mistralai.search.toolkit.embedding import DistanceMetric
from mistralai.search.toolkit.search import SearchResult, SearchResultChunk
from pydantic import BaseModel
from qdrant_client import models

from mistralai.search.toolkit.plugins.qdrant.schema import CustomFieldMapping


def point_id(chunk_id: str) -> str:
    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def qdrant_distance(metric: DistanceMetric) -> models.Distance:
    if metric == DistanceMetric.L2:
        return models.Distance.EUCLID
    if metric == DistanceMetric.INNER_PRODUCT:
        return models.Distance.DOT
    return models.Distance.COSINE


def payload_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [payload_value(item) for item in value]
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def to_payload(
    document: Document, chunk: DocumentChunk, custom: Iterable[CustomFieldMapping]
) -> dict[str, Any]:
    blob = chunk.metadata.model_dump(mode="json", exclude_none=True)
    for key, value in document.metadata.model_dump(
        mode="json", exclude_none=True
    ).items():
        blob[f"document_{key}"] = value
    payload: dict[str, Any] = {
        "id": chunk.id,
        "document_id": document.id,
        "source_id": chunk.source_id,
        "locator": chunk.locator,
        "parent_ref": chunk.parent_ref,
        "chunk_type": chunk.chunk_type.value,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "content": chunk.content,
        "metadata": blob,
    }
    for field in custom:
        src = document if field.source == "document" else chunk
        payload[field.key] = payload_value(getattr(src, field.field, None))
    return payload


def to_chunk(
    payload: Mapping[str, Any],
    custom_keys: Iterable[str],
    include_content: bool = True,
    include_metadata: bool = True,
) -> SearchResultChunk:
    return SearchResultChunk(
        id=str(payload["id"]),
        source_id=str(payload["source_id"]),
        locator=str(payload["locator"]),
        parent_ref=payload.get("parent_ref"),
        start_offset=int(payload["start_offset"]),
        end_offset=int(payload["end_offset"]),
        chunk_type=payload["chunk_type"],
        content=str(payload.get("content") or "") if include_content else "",
        metadata=dict(payload["metadata"])
        if include_metadata and payload.get("metadata")
        else {},
        **{k: payload[k] for k in custom_keys if k in payload},
    )


def distance_to_score(metric: DistanceMetric, distance: float) -> float:
    if metric == DistanceMetric.COSINE:
        return 1.0 - distance
    if metric == DistanceMetric.L2:
        return 1.0 / (1.0 + distance)
    return -distance


def score_to_distance(metric: DistanceMetric, score: float) -> float:
    if metric == DistanceMetric.L2:
        return score
    if metric == DistanceMetric.INNER_PRODUCT:
        return -score
    return 1.0 - score


def ranked(
    payload: Mapping[str, Any],
    score: float,
    metric: DistanceMetric,
    custom_keys: Iterable[str],
    include_content: bool,
    include_metadata: bool,
) -> SearchResult:
    distance = score_to_distance(metric, float(score))
    return SearchResult(
        chunk=to_chunk(payload, custom_keys, include_content, include_metadata),
        score=distance_to_score(metric, distance),
        distance=distance,
    )


def unranked(payload: Mapping[str, Any], custom_keys: Iterable[str]) -> SearchResult:
    return SearchResult(chunk=to_chunk(payload, custom_keys), score=0.0, distance=None)


def fuse(
    dense: list[tuple[str, float]],
    lexical: list[str],
    payloads: Mapping[str, Mapping[str, Any]],
    metric: DistanceMetric,
    custom_keys: Iterable[str],
    *,
    top_k: int,
    vector_weight: float,
    text_weight: float,
    rrf_k: int,
    include_content: bool,
    include_metadata: bool,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    for rank, (cid, _) in enumerate(dense, start=1):
        scores[cid] = scores.get(cid, 0.0) + vector_weight / (rrf_k + rank)
    for rank, cid in enumerate(lexical, start=1):
        scores[cid] = scores.get(cid, 0.0) + text_weight / (rrf_k + rank)
    dense_raw = dict(dense)
    keys = list(custom_keys)
    return [
        SearchResult(
            chunk=to_chunk(payloads[cid], keys, include_content, include_metadata),
            score=scores[cid],
            distance=score_to_distance(metric, dense_raw[cid])
            if cid in dense_raw
            else None,
        )
        for cid in sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    ]
