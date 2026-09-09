# mistralai-search-toolkit-plugins-qdrant

[Qdrant](https://qdrant.tech/) plugin for [Mistral Search Toolkit](https://docs.mistral.ai/studio/search/search-toolkit).

Requires Python 3.12+.

## Install

```bash
uv add mistralai-search-toolkit-plugins-qdrant
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

```python
from mistralai.search.toolkit.document import Document, DocumentChunk
from mistralai.search.toolkit.embedding import CustomEmbeddingModel
from mistralai.search.toolkit.plugins.qdrant import (
    QdrantApp,
    QdrantCollectionSchema,
    QdrantConnectionConfig,
)
from mistralai.search.toolkit.search import VectorSearchQuery

DIM = 1024
schema = QdrantCollectionSchema(
    collection_name="docs",
    document_type=Document,
    embedding_model=CustomEmbeddingModel(name="my-embedder", dimensions=DIM),
)
app = QdrantApp([schema])
config = QdrantConnectionConfig(url="http://localhost:6333")

await app.create_collection(config, "docs")
store = app.get_search_index(config, "docs")

doc = Document(
    source_id="notes.md",
    content="quarterly revenue grew",
    chunks=[
        DocumentChunk(
            source_id="notes.md",
            locator="char:0-22",
            start_offset=0,
            end_offset=22,
            content="quarterly revenue grew",
            embedding=[0.1] * DIM,
        )
    ],
)
await store.index_document(doc)
hits = await store.search(VectorSearchQuery(embedding=[0.1] * DIM, top_k=10))
await store.aclose()
```

`create_collection` is safe to run twice. Pass an `AsyncQdrantClient` instead of a config if you already have one. Then `aclose()` leaves it open.

Qdrant Cloud: `QdrantConnectionConfig(url="https://....qdrant.io:6333", api_key="...")`.

## Search

Add text to the query for hybrid search (vector + full-text on `content`, fused with RRF):

```python
from mistralai.search.toolkit.plugins.qdrant import QdrantSearchQuery

await store.search(
    QdrantSearchQuery(embedding=vec, query="quarterly revenue", top_k=10)
)
```

`exclude_ids` skips chunks. `max_candidates` is Qdrant `hnsw_ef`. Higher `score` is better. For cosine, `distance` is `1 - score`. Hybrid `score` only ranks inside that one result list. Text-only hits have `distance=None`.

Walk a document with `navigate`, `read`, `grep`, `get_chunk`. Offsets are `[start, end)`. `PHRASE` grep wants the words in order. `TERM` wants every word, any order.

Patch without reindexing:

```python
from mistralai.search.toolkit.document import ChunkPatch, DocumentPatch

await store.patch_chunk(chunk_id, ChunkPatch(content="new text"))
await store.patch_document(doc.id, DocumentPatch(metadata={"status": "published"}))
```

Set a metadata key to `None` to drop it.

## Extra fields

```python
from typing import Annotated
from mistralai.search.toolkit.document import Document, DocumentChunk
from mistralai.search.toolkit.plugins.qdrant import QdrantField


class ArticleChunk(DocumentChunk):
    section: Annotated[str | None, QdrantField()] = None


class Article(Document):
    title: Annotated[str | None, QdrantField(name="headline")] = None
```

Use `document_type=Article`. Chunk fields keep their name. Document fields get a `document_` prefix unless you rename them. `QdrantField(ignore=True)` skips a field.

Qdrant point ids are integers or UUIDs. Normal UUID5 chunk ids go through as-is. Anything else is hashed. The original string stays in payload `id`.

## Tests

Needs Qdrant on `localhost:6333` (or set `QDRANT_URL`).

```bash
uv sync --group dev && uv run pytest tests/ -q
```
