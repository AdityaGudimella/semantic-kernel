# Copyright (c) Microsoft. All rights reserved.

import pydantic as pdt

from semantic_kernel.memory.memory_record import MemoryRecord


class MemoryQueryResult(MemoryRecord):
    relevance: float = pdt.Field(
        description="The relevance of the record to a known query.",
    )

    @classmethod
    def from_memory_record(
        cls,
        record: MemoryRecord,
        relevance: float,
    ) -> "MemoryQueryResult":
        """Create a new instance of MemoryQueryResult from a MemoryRecord.

        Arguments:
            record {MemoryRecord} -- The MemoryRecord to create the MemoryQueryResult from.
            relevance {float} -- The relevance of the record to a known query.

        Returns:
            MemoryQueryResult -- The created MemoryQueryResult.
        """
        return cls(relevance=relevance, **record.dict())
