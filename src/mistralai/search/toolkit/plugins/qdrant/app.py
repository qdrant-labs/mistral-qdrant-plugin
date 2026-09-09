from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict
from qdrant_client import AsyncQdrantClient, models

from mistralai.search.toolkit.plugins.qdrant.index import QdrantStoreIndex
from mistralai.search.toolkit.plugins.qdrant.mapping import qdrant_distance
from mistralai.search.toolkit.plugins.qdrant.schema import QdrantCollectionSchema

_INDEXES: tuple[tuple[str, models.PayloadSchemaType | models.TextIndexParams], ...] = (
    ("id", models.PayloadSchemaType.KEYWORD),
    ("document_id", models.PayloadSchemaType.KEYWORD),
    ("source_id", models.PayloadSchemaType.KEYWORD),
    ("locator", models.PayloadSchemaType.KEYWORD),
    ("chunk_type", models.PayloadSchemaType.KEYWORD),
    ("start_offset", models.PayloadSchemaType.INTEGER),
    ("end_offset", models.PayloadSchemaType.INTEGER),
    (
        "content",
        models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
            phrase_matching=True,
        ),
    ),
)


class QdrantConnectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str = "http://localhost:6333"
    api_key: str | None = None
    timeout: int = 60
    prefer_grpc: bool = False

    def client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout,
            prefer_grpc=self.prefer_grpc,
        )


class QdrantApp:
    def __init__(self, schemas: Iterable[QdrantCollectionSchema]) -> None:
        self._schemas: dict[str, QdrantCollectionSchema] = {}
        for schema in schemas:
            if schema.collection_name in self._schemas:
                raise ValueError(
                    f"collection {schema.collection_name!r} is declared more than once"
                )
            self._schemas[schema.collection_name] = schema

    def collection(self, name: str) -> QdrantCollectionSchema:
        try:
            return self._schemas[name]
        except KeyError:
            raise ValueError(
                f"unknown collection {name!r}; known: {sorted(self._schemas)}"
            ) from None

    def get_search_index(
        self, src: QdrantConnectionConfig | AsyncQdrantClient, collection_name: str
    ) -> QdrantStoreIndex:
        schema = self.collection(collection_name)
        if isinstance(src, QdrantConnectionConfig):
            return QdrantStoreIndex(src.client(), schema, owns_client=True)
        return QdrantStoreIndex(src, schema)

    async def create_collection(
        self, src: QdrantConnectionConfig | AsyncQdrantClient, collection_name: str
    ) -> None:
        schema = self.collection(collection_name)
        owns = isinstance(src, QdrantConnectionConfig)
        client = src.client() if owns else src
        try:
            if not await client.collection_exists(collection_name=collection_name):
                model = schema.embedding_model
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=model.dimensions,
                        distance=qdrant_distance(model.distance_metric),
                    ),
                )
            for field, field_schema in _INDEXES:
                await client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=field_schema,
                )
        finally:
            if owns:
                await client.close()
