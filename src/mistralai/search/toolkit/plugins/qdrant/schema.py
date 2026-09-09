from dataclasses import dataclass
from typing import Literal, get_args

from mistralai.search.toolkit.document import Document, DocumentChunk
from mistralai.search.toolkit.embedding import EmbeddingModel, MistralEmbeddingPreset


@dataclass(frozen=True)
class QdrantField:
    name: str | None = None
    ignore: bool = False


@dataclass(frozen=True)
class CustomFieldMapping:
    key: str
    field: str
    source: Literal["chunk", "document"]


def custom_field_mappings(document_type: type[Document]) -> list[CustomFieldMapping]:
    chunk_type = DocumentChunk
    args = get_args(document_type.model_fields["chunks"].annotation)
    if args:
        element = args[0]
        if hasattr(element, "__metadata__"):
            element = get_args(element)[0]
        if isinstance(element, type) and issubclass(element, DocumentChunk):
            chunk_type = element

    mappings: list[CustomFieldMapping] = []
    for model, base, source, prefix in (
        (document_type, Document, "document", "document_"),
        (chunk_type, DocumentChunk, "chunk", ""),
    ):
        for name in sorted(model.model_fields.keys() - base.model_fields.keys()):
            override = next(
                (
                    m
                    for m in model.model_fields[name].metadata
                    if isinstance(m, QdrantField)
                ),
                QdrantField(),
            )
            if not override.ignore:
                mappings.append(
                    CustomFieldMapping(
                        key=override.name or f"{prefix}{name}",
                        field=name,
                        source=source,
                    )
                )
    return mappings


@dataclass(frozen=True)
class QdrantCollectionSchema:
    collection_name: str
    document_type: type[Document]
    embedding_model: EmbeddingModel | MistralEmbeddingPreset

    def __post_init__(self) -> None:
        model = (
            self.embedding_model.build_embedding_model()
            if isinstance(self.embedding_model, MistralEmbeddingPreset)
            else self.embedding_model
        )
        object.__setattr__(self, "embedding_model", model)
        if not self.collection_name.strip():
            raise ValueError("collection_name must be a non-empty string")

    def custom_mappings(self) -> list[CustomFieldMapping]:
        return custom_field_mappings(self.document_type)
