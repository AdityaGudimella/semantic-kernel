# Copyright (c) Microsoft. All rights reserved.

import asyncio
import typing as t

import numpy as np
import pydantic as pdt
import weaviate
from pydantic.utils import GetterDict
from weaviate.embedded import EmbeddedOptions

from semantic_kernel.logging_ import NullLogger, SKLogger
from semantic_kernel.memory.memory_record import MemoryRecord
from semantic_kernel.memory.memory_store_base import MemoryStoreBase
from semantic_kernel.pydantic_ import SKBaseModel

_ALL_MEMORY_RECORD_PROPERTIES: t.Final[t.List[str]] = list(
    MemoryRecord.schema()["properties"]
)


class WeaviateConfig(SKBaseModel):
    use_embed: bool = False
    url: t.Optional[str] = None
    api_key: t.Optional[str] = None


class MemoryRecordMapper(GetterDict):
    """This class is used to define the mapping between `MemoryRecord` and Weaviate."""

    _MAPPING: t.Final[t.Dict[str, str]] = {
        # _WeaviateRecord.vector maps to MemoryRecord.embedding
        "vector": "embedding",
        "embedding": "vector",
    }

    def get(self, key: str, default: t.Any) -> t.Any:
        if key in self._MAPPING:
            key = self._MAPPING[key]
        return super().get(key, default)


class _WeaviateRecord(SKBaseModel):
    key: t.Optional[str] = None
    timestamp: t.Optional[str] = None
    is_reference: t.Optional[bool] = None
    external_source_name: t.Optional[str] = None
    id_: t.Optional[str] = pdt.Field(alias="skId", default=None)
    description: t.Optional[str] = None
    text: str
    additional_metadata: t.Optional[str]
    vector: t.Optional[t.List[float]] = pdt.Field(
        alias="embedding", description="The embedding of the record."
    )

    class Config:
        """Pydantic config."""

        orm_mode = True
        getter_dict = MemoryRecordMapper

    @pdt.validator("vector", pre=True)
    def _validate_vector(cls, v: t.Any):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if not isinstance(v, list):
            raise ValueError("vector must be a list or a numpy array")
        return v

    def to_memory_record(self) -> MemoryRecord:
        """Convert to a MemoryRecord."""
        return MemoryRecord(**self.dict(by_alias=True))


class WeaviateMemoryStore(SKBaseModel, MemoryStoreBase):
    config: WeaviateConfig
    _client: t.Optional[weaviate.Client] = pdt.PrivateAttr(default=None)
    _logger: SKLogger = pdt.PrivateAttr(default_factory=NullLogger)

    @property
    def client(self):
        if self._client is None:
            self._client = self._initialize_client(self.config)
        return self._client

    @staticmethod
    def _initialize_client(config):
        if config.use_embed:
            return weaviate.Client(embedded_options=EmbeddedOptions())
        if not config.url:
            raise ValueError("Weaviate config must have either url or use_embed set")
        if config.api_key:
            return weaviate.Client(
                url=config.url,
                auth_client_secret=weaviate.auth.AuthApiKey(api_key=config.api_key),
            )
        else:
            return weaviate.Client(url=config.url)

    async def create_collection_async(self, collection_name: str) -> None:
        schema = MemoryRecord.schema()
        schema["class"] = collection_name
        await asyncio.get_running_loop().run_in_executor(
            None, self.client.schema.create_class, schema
        )

    async def get_collections_async(self) -> t.List[str]:
        schemas = await asyncio.get_running_loop().run_in_executor(
            None, self.client.schema.get
        )
        return [schema["class"] for schema in schemas["classes"]]

    async def delete_collection_async(self, collection_name: str) -> bool:
        await asyncio.get_running_loop().run_in_executor(
            None, self.client.schema.delete_class, collection_name
        )

    async def does_collection_exist_async(self, collection_name: str) -> bool:
        collections = await self.get_collections_async()
        return collection_name in collections

    async def upsert_async(self, collection_name: str, record: MemoryRecord) -> str:
        weaviate_record = _WeaviateRecord.from_orm(record).dict()

        vector = weaviate_record.pop("vector", None)
        weaviate_id = weaviate.util.generate_uuid5(weaviate_record, collection_name)

        return await asyncio.get_running_loop().run_in_executor(
            None,
            self.client.data_object.create,
            weaviate_record,
            collection_name,
            weaviate_id,
            vector,
        )

    async def upsert_batch_async(
        self, collection_name: str, records: t.List[MemoryRecord]
    ) -> t.List[str]:
        def _upsert_batch_inner():
            results = []
            with self.client.batch as batch:
                for record in records:
                    weaviate_record = _WeaviateRecord.from_orm(record).dict()
                    vector = weaviate_record.pop("vector", None)
                    weaviate_id = weaviate.util.generate_uuid5(
                        weaviate_record, collection_name
                    )
                    batch.add_data_object(
                        data_object=weaviate_record,
                        uuid=weaviate_id,
                        vector=vector,
                        class_name=collection_name,
                    )
                    results.append(weaviate_id)

            return results

        return await asyncio.get_running_loop().run_in_executor(
            None, _upsert_batch_inner
        )

    async def get_async(
        self, collection_name: str, key: str, with_embedding: bool
    ) -> MemoryRecord:
        # Call the batched version with a single key
        results = await self.get_batch_async(collection_name, [key], with_embedding)
        return results[0] if results else None

    async def get_batch_async(
        self, collection_name: str, keys: t.List[str], with_embedding: bool
    ) -> t.List[MemoryRecord]:
        queries = self._build_multi_get_query(collection_name, keys, with_embedding)

        results = await asyncio.get_running_loop().run_in_executor(
            None, self.client.query.multi_get(queries).do
        )

        get_dict = results.get("data", {}).get("Get", {})

        return [
            self._convert_weaviate_doc_to_memory_record(doc)
            for docs in get_dict.values()
            for doc in docs
        ]

    def _build_multi_get_query(
        self, collection_name: str, keys: t.List[str], with_embedding: bool
    ):
        queries = []
        for i, key in enumerate(keys):
            query = self.client.query.get(
                collection_name, _ALL_MEMORY_RECORD_PROPERTIES
            ).with_where(
                {
                    "path": ["key"],
                    "operator": "Equal",
                    "valueString": key,
                }
            )
            if with_embedding:
                query = query.with_additional("vector")

            query = query.with_alias(f"query_{i}")

            queries.append(query)

        return queries

    def _convert_weaviate_doc_to_memory_record(
        self, weaviate_doc: dict
    ) -> MemoryRecord:
        weaviate_doc_copy = weaviate_doc.copy()
        vector = weaviate_doc_copy.pop("_additional", {}).get("vector")
        weaviate_doc_copy["vector"] = np.array(vector) if vector else None
        return MemoryRecord.from_orm(_WeaviateRecord(**weaviate_doc_copy))

    async def remove_async(self, collection_name: str, key: str) -> None:
        await self.remove_batch_async(collection_name, [key])

    async def remove_batch_async(self, collection_name: str, keys: t.List[str]) -> None:
        # TODO: Use In operator when it's available
        #       (https://github.com/weaviate/weaviate/issues/2387)
        #       and handle max delete objects
        #       (https://weaviate.io/developers/weaviate/api/rest/batch#maximum-number-of-deletes-per-query)
        for key in keys:
            where = {
                "path": ["key"],
                "operator": "Equal",
                "valueString": key,
            }

            await asyncio.get_running_loop().run_in_executor(
                None, self.client.batch.delete_objects, collection_name, where
            )

    async def get_nearest_matches_async(
        self,
        collection_name: str,
        embedding: np.ndarray,
        limit: int,
        min_relevance_score: float,
        with_embeddings: bool,
    ) -> t.List[t.Tuple[MemoryRecord, float]]:
        nearVector = {
            "vector": embedding,
            "certainty": min_relevance_score,
        }

        additional = ["certainty"]
        if with_embeddings:
            additional.append("vector")

        query = (
            self.client.query.get(collection_name, _ALL_MEMORY_RECORD_PROPERTIES)
            .with_near_vector(nearVector)
            .with_additional(additional)
            .with_limit(limit)
        )

        results = await asyncio.get_running_loop().run_in_executor(None, query.do)

        get_dict = results.get("data", {}).get("Get", {})

        return [
            (
                self._convert_weaviate_doc_to_memory_record(doc),
                item["_additional"]["certainty"],
            )
            for items in get_dict.values()
            for item in items
            # TODO(ADI): Why the `for doc in [item]`?
            for doc in [item]
        ]

    async def get_nearest_match_async(
        self,
        collection_name: str,
        embedding: np.ndarray,
        min_relevance_score: float,
        with_embedding: bool,
    ) -> t.Tuple[MemoryRecord, float]:
        results = await self.get_nearest_matches_async(
            collection_name,
            embedding,
            limit=1,
            min_relevance_score=min_relevance_score,
            with_embeddings=with_embedding,
        )

        return results[0]
