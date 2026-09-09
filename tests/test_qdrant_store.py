import os
import uuid

import pytest
from mistralai.search.toolkit.document import (
    ChunkPatch,
    ChunkType,
    Document,
    DocumentChunk,
    DocumentPatch,
)
from mistralai.search.toolkit.embedding import CustomEmbeddingModel, DistanceMetric
from mistralai.search.toolkit.search import (
    ChunkNotFoundError,
    DocumentNotFoundError,
    GrepMode,
    NavigationDirection,
    SearchError,
    VectorSearchQuery,
)
from mistralai.search.toolkit.search.errors import IndexingError, SourceNotFoundError
from qdrant_client import AsyncQdrantClient

from mistralai.search.toolkit.plugins.qdrant import (
    QdrantApp,
    QdrantCollectionSchema,
    QdrantSearchQuery,
)

DIM = 8
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _vec(seed: float) -> list[float]:
    return [seed + i * 0.01 for i in range(DIM)]


def _doc(source_id: str = "doc.md", n: int = 3, doc_id: str | None = None) -> Document:
    chunks = [
        DocumentChunk(
            source_id=source_id,
            locator=f"char:{i * 10}-{(i + 1) * 10}",
            start_offset=i * 10,
            end_offset=(i + 1) * 10,
            content=f"chunk {i} about revenue and markets",
            embedding=_vec(float(i)),
        )
        for i in range(n)
    ]
    return Document(
        id=doc_id or str(uuid.uuid5(uuid.NAMESPACE_DNS, source_id)),
        source_id=source_id,
        content="full",
        chunks=chunks,
    )


async def _qdrant():
    client = AsyncQdrantClient(url=QDRANT_URL, timeout=5)
    try:
        await client.info()
    except Exception:
        await client.close()
        pytest.skip(f"Qdrant is not running at {QDRANT_URL}")
    return client


def _schema(name: str, metric=DistanceMetric.COSINE, document_type=Document):
    return QdrantCollectionSchema(
        collection_name=name,
        document_type=document_type,
        embedding_model=CustomEmbeddingModel(
            name="t", dimensions=DIM, distance_metric=metric
        ),
    )


@pytest.fixture()
async def store():
    from mistralai.search.toolkit.plugins.qdrant.index import QdrantStoreIndex

    name = f"docs_{uuid.uuid4().hex[:8]}"
    client = await _qdrant()
    app = QdrantApp([_schema(name)])
    await app.create_collection(client, name)
    s = app.get_search_index(client, name)
    assert isinstance(s, QdrantStoreIndex)
    try:
        yield s
    finally:
        await client.delete_collection(collection_name=name)
        await client.close()


async def test_index_search_delete(store):
    doc = _doc()
    await store.index_document(doc)
    results = await store.search(VectorSearchQuery(embedding=_vec(0.0), top_k=3))
    assert len(results) == 3
    assert results[0].chunk.source_id == "doc.md"
    assert results[0].score is not None
    assert results[0].score == pytest.approx(1.0, abs=1e-3)
    assert results[0].distance == pytest.approx(0.0, abs=1e-3)
    for r in results:
        assert r.distance == pytest.approx(1.0 - r.score, abs=1e-6)

    await store.index_document(doc)
    results = await store.search(VectorSearchQuery(embedding=_vec(0.0), top_k=10))
    assert len(results) == 3

    await store.delete_document(doc.id)
    with pytest.raises(DocumentNotFoundError):
        await store.delete_document(doc.id)


async def test_index_validation(store):
    with pytest.raises(IndexingError):
        await store.index_document(Document(source_id="x", content="empty", chunks=[]))
    bad = Document(
        source_id="x",
        content="no embeddings",
        chunks=[
            DocumentChunk(
                source_id="x",
                locator="char:0-5",
                start_offset=0,
                end_offset=5,
                content="hello",
            ),
        ],
    )
    with pytest.raises(IndexingError):
        await store.index_document(bad)
    with pytest.raises(SearchError):
        await store.search(VectorSearchQuery(embedding=_vec(0.0), top_k=0))


async def test_exclude_ids_and_top_k(store):
    doc = _doc()
    await store.index_document(doc)
    all_results = await store.search(VectorSearchQuery(embedding=_vec(0.0), top_k=3))
    excluded = {all_results[0].chunk.id}
    rest = await store.search(
        VectorSearchQuery(embedding=_vec(0.0), top_k=3, exclude_ids=excluded)
    )
    assert {r.chunk.id for r in rest} == {r.chunk.id for r in all_results} - excluded


async def test_hybrid_search(store):
    doc = _doc()
    await store.index_document(doc)
    results = await store.search(
        VectorSearchQuery(embedding=_vec(0.0), query="revenue markets", top_k=3)
    )
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(r.distance is not None for r in results)

    tuned = await store.search(
        QdrantSearchQuery(
            embedding=_vec(0.0),
            query="revenue markets",
            top_k=3,
            text_weight=5.0,
            rrf_k=60,
        )
    )
    assert len(tuned) == 3


async def test_euclid_distances():
    name = f"euclid_{uuid.uuid4().hex[:8]}"
    client = await _qdrant()
    app = QdrantApp([_schema(name, DistanceMetric.L2)])
    await app.create_collection(client, name)
    store = app.get_search_index(client, name)
    try:
        await store.index_document(_doc())
        results = await store.search(VectorSearchQuery(embedding=_vec(0.0), top_k=3))
        assert results[0].distance == pytest.approx(0.0, abs=1e-3)
        for r in results:
            assert r.score == pytest.approx(1.0 / (1.0 + r.distance), abs=1e-6)
    finally:
        await client.delete_collection(collection_name=name)
        await client.close()


async def test_navigate_read_get_chunk(store):
    doc = _doc(n=5)
    await store.index_document(doc)

    nxt = await store.navigate("doc.md", 0, 10, NavigationDirection.NEXT, top_k=2)
    assert [c.chunk.start_offset for c in nxt] == [10, 20]

    prev = await store.navigate("doc.md", 30, 40, NavigationDirection.PREVIOUS, top_k=2)
    assert [c.chunk.start_offset for c in prev] == [10, 20]

    window = await store.read("doc.md", 10, 30)
    assert [c.chunk.start_offset for c in window] == [10, 20]

    everything = await store.read("doc.md", None, None)
    assert len(everything) == 5

    chunk = await store.get_chunk(nxt[0].chunk.id)
    assert chunk is not None and chunk.chunk.id == nxt[0].chunk.id
    assert chunk.score == 0.0
    assert await store.get_chunk("does-not-exist") is None

    with pytest.raises(SourceNotFoundError):
        await store.navigate("missing.md", 0, 10, NavigationDirection.NEXT)
    with pytest.raises(SourceNotFoundError):
        await store.read("missing.md", 0, 10)


async def test_grep(store):
    chunks = [
        DocumentChunk(
            source_id="g.md",
            locator="char:0-10",
            start_offset=0,
            end_offset=10,
            content="the quick brown fox",
            embedding=_vec(0.0),
        ),
        DocumentChunk(
            source_id="g.md",
            locator="char:10-20",
            start_offset=10,
            end_offset=20,
            content="jumps over the lazy dog",
            embedding=_vec(1.0),
        ),
    ]
    await store.index_document(
        Document(source_id="g.md", content="full", chunks=chunks)
    )
    hits = await store.grep("g.md", "quick brown", mode=GrepMode.PHRASE)
    assert len(hits) == 1 and hits[0].chunk.start_offset == 0
    hits = await store.grep("g.md", "fox dog", mode=GrepMode.TERM)
    assert await store.grep("g.md", "   ") == []
    with pytest.raises(SourceNotFoundError):
        await store.grep("missing.md", "fox")


async def test_patch(store):
    doc = _doc()
    await store.index_document(doc)
    chunk_id = (await store.search(VectorSearchQuery(embedding=_vec(0.0), top_k=1)))[
        0
    ].chunk.id

    await store.patch_chunk(chunk_id, ChunkPatch(content="updated content"))
    assert (await store.get_chunk(chunk_id)).chunk.content == "updated content"

    await store.patch_chunk(chunk_id, ChunkPatch(metadata={"k": "v"}))
    assert (await store.get_chunk(chunk_id)).chunk.metadata["k"] == "v"

    await store.patch_document(doc.id, DocumentPatch(metadata={"dk": "dv"}))
    assert (await store.get_chunk(chunk_id)).chunk.metadata["document_dk"] == "dv"

    with pytest.raises(ChunkNotFoundError):
        await store.patch_chunk("missing-chunk", ChunkPatch(content="x"))
    with pytest.raises(DocumentNotFoundError):
        await store.patch_document("missing-doc", DocumentPatch())


async def test_custom_fields():
    from typing import Annotated

    from mistralai.search.toolkit.plugins.qdrant import QdrantField

    class MyChunk(DocumentChunk):
        section: Annotated[str | None, QdrantField()] = None

    class MyDoc(Document):
        chunks: list[MyChunk] = []
        title: Annotated[str | None, QdrantField(name="headline")] = None

    name = f"custom_{uuid.uuid4().hex[:8]}"
    client = await _qdrant()
    app = QdrantApp([_schema(name, document_type=MyDoc)])
    await app.create_collection(client, name)
    store = app.get_search_index(client, name)
    chunk = MyChunk(
        source_id="c.md",
        locator="char:0-5",
        start_offset=0,
        end_offset=5,
        content="hello",
        embedding=_vec(0.0),
        section="intro",
    )
    await store.index_document(
        MyDoc(source_id="c.md", content="hello", chunks=[chunk], title="T")
    )
    results = await store.search(VectorSearchQuery(embedding=_vec(0.0), top_k=1))
    assert results[0].chunk.section == "intro"
    assert results[0].chunk.headline == "T"
    await client.delete_collection(collection_name=name)
    await client.close()


async def test_chunk_type_filtering(store):
    chunks = [
        DocumentChunk(
            source_id="m.md",
            locator="char:0-5",
            start_offset=0,
            end_offset=5,
            content="main",
            embedding=_vec(0.0),
            chunk_type=ChunkType.CONTENT,
        ),
        DocumentChunk(
            source_id="m.md",
            locator="summary:char:0-5",
            start_offset=0,
            end_offset=5,
            content="summary",
            embedding=_vec(0.0),
            chunk_type=ChunkType.SUMMARY,
        ),
    ]
    await store.index_document(
        Document(source_id="m.md", content="full", chunks=chunks)
    )
    assert len(await store.read("m.md", None, None)) == 1
    assert (
        len(await store.read("m.md", None, None, content_type=ChunkType.SUMMARY)) == 1
    )


async def test_content_index_enables_phrase_match():
    name = f"phr_{uuid.uuid4().hex[:8]}"
    client = await _qdrant()
    app = QdrantApp([_schema(name)])
    await app.create_collection(client, name)
    store = app.get_search_index(client, name)
    try:
        chunks = [
            DocumentChunk(
                source_id="g.md",
                locator="char:0-10",
                start_offset=0,
                end_offset=10,
                content="the quick brown fox",
                embedding=_vec(0.0),
            ),
            DocumentChunk(
                source_id="g.md",
                locator="char:10-20",
                start_offset=10,
                end_offset=20,
                content="jumps over the lazy dog",
                embedding=_vec(1.0),
            ),
        ]
        await store.index_document(
            Document(source_id="g.md", content="full", chunks=chunks)
        )
        hits = await store.grep("g.md", "quick brown", mode=GrepMode.PHRASE)
        assert len(hits) == 1 and hits[0].chunk.start_offset == 0
    finally:
        await client.delete_collection(collection_name=name)
        await client.close()


async def test_patch_document_preserves_vectors(store):
    doc = _doc()
    await store.index_document(doc)
    before = await store.search(VectorSearchQuery(embedding=_vec(0.0), top_k=3))
    await store.patch_document(doc.id, DocumentPatch(metadata={"dk": "dv"}))
    after = await store.search(VectorSearchQuery(embedding=_vec(0.0), top_k=3))
    assert [(r.chunk.id, r.score) for r in after] == [
        (r.chunk.id, r.score) for r in before
    ]
    assert all(r.distance is not None for r in after)
