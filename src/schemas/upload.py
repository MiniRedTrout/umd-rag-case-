from typing import Literal, Optional

from pydantic import BaseModel, Field


class ImportResult(BaseModel):
    """Base pydantic model for the import endpoint's answer."""

    status: Literal["success", "processing", "error"] = Field(
        default=None,
        description="The result of the data import request",
    )
    message: str = Field(
        default=None,
        description="Information about the import",
    )
    total_documents: Optional[int] = Field(
        default=None,
        description="Number of documents added to the system",
    )