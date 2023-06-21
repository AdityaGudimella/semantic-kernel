# Copyright (c) Microsoft. All rights reserved.

import typing as t

import pydantic as pdt
from numpy import ndarray

from semantic_kernel.pydantic_ import PydanticNDArray, SKBaseModel

SKBaseModel.from_orm


class MemoryRecord(SKBaseModel):
    """A record in the memory."""

    key: t.Optional[str] = pdt.Field(
        default=None,
        description="The key of the record.",
    )
    timestamp: t.Optional[str] = pdt.Field(
        default=None,
        description="The timestamp of the record.",
    )
    is_reference: t.Optional[bool] = pdt.Field(
        description="Whether the record is a reference record.",
    )
    # TODO(ADI): Why don't these optional fields haved default values?
    external_source_name: t.Optional[str] = pdt.Field(
        description="The name of the external source.",
    )
    id_: str = pdt.Field(
        alias="skId",
        description="A unique for the record.",
    )
    description: t.Optional[str] = pdt.Field(
        description="The description of the record.",
    )
    text: t.Optional[str] = pdt.Field(
        description="The text of the record.",
    )
    additional_metadata: t.Optional[str] = pdt.Field(
        description="Custom metadata for the record.",
    )
    embedding: t.Optional[PydanticNDArray] = pdt.Field(
        description="The embedding of the record.",
    )

    @staticmethod
    def reference_record(
        external_id: str,
        source_name: str,
        description: t.Optional[str],
        additional_metadata: t.Optional[str],
        embedding: ndarray,
    ) -> "MemoryRecord":
        """Create a reference record.

        Arguments:
            external_id {str} -- The external id of the record.
            source_name {str} -- The name of the external source.
            description {Optional[str]} -- The description of the record.
            additional_metadata {Optional[str]} -- Custom metadata for the record.
            embedding {ndarray} -- The embedding of the record.

        Returns:
            MemoryRecord -- The reference record.
        """
        return MemoryRecord(
            is_reference=True,
            external_source_name=source_name,
            id_=external_id,
            description=description,
            text=None,
            additional_metadata=additional_metadata,
            embedding=embedding,
        )

    @staticmethod
    def local_record(
        id: str,
        text: str,
        description: t.Optional[str],
        additional_metadata: t.Optional[str],
        embedding: ndarray,
    ) -> "MemoryRecord":
        """Create a local record.

        Arguments:
            id {str} -- A unique for the record.
            text {str} -- The text of the record.
            description {Optional[str]} -- The description of the record.
            additional_metadata {Optional[str]} -- Custom metadata for the record.
            embedding {ndarray} -- The embedding of the record.

        Returns:
            MemoryRecord -- The local record.
        """
        return MemoryRecord(
            is_reference=False,
            external_source_name=None,
            id_=id,
            description=description,
            text=text,
            additional_metadata=additional_metadata,
            embedding=embedding,
        )
