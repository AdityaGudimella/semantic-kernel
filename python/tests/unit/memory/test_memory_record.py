import typing as t

import pytest

from semantic_kernel.memory.memory_record import MemoryRecord


@pytest.fixture
def memory_record() -> MemoryRecord:
    return MemoryRecord(
        key="key",
        timestamp="timestamp",
        is_reference=True,
        external_source_name="external_source_name",
        id_="id_",
        description="description",
        text="text",
        additional_metadata="additional_metadata",
        embedding=[1, 2, 3],
    )


@pytest.fixture()
def original_schema() -> dict:
    """SCHEMA defined in `weaviate_memory_store.py`"""
    return {
        "class": "MemoryRecord",
        "description": "A document from semantic kernel.",
        "properties": [
            {
                "name": "key",
                "description": "The key of the record.",
                "dataType": ["string"],
            },
            {
                "name": "timestamp",
                "description": "The timestamp of the record.",
                "dataType": ["date"],
            },
            {
                "name": "isReference",
                "description": "Whether the record is a reference record.",
                "dataType": ["boolean"],
            },
            {
                "name": "externalSourceName",
                "description": "The name of the external source.",
                "dataType": ["string"],
            },
            {
                "name": "skId",
                "description": "A unique identifier for the record.",
                "dataType": ["string"],
            },
            {
                "name": "description",
                "description": "The description of the record.",
                "dataType": ["text"],
            },
            {
                "name": "text",
                "description": "The text of the record.",
                "dataType": ["text"],
            },
            {
                "name": "additionalMetadata",
                "description": "Optional custom metadata of the record.",
                "dataType": ["string"],
            },
        ],
    }


@pytest.fixture()
def original_all_properties(original_schema: dict) -> t.List[str]:
    """All properties defined in `weaviate_memory_store.py`"""
    return [property_["name"] for property_ in original_schema["properties"]]


@pytest.mark.xfail(reason="Original schema does not contain embedding.")
def test_schema(
    memory_record: MemoryRecord, original_all_properties: t.List[str]
) -> None:
    """Ensure the schema is created correctly."""
    assert (
        list(memory_record.schema(by_alias=True)["properties"])
        == original_all_properties
    )
