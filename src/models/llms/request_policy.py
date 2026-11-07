"""Declare gateway routing metadata for chat requests."""

import os
from dataclasses import dataclass
from enum import StrEnum

CONTENT_ORIGIN_HEADER = "X-Content-Origin"
SERVICE_NAME_HEADER = "X-Service-Name"
DEFAULT_SERVICE_NAME = "spok-case-nmd"


class InputProvenance(StrEnum):
    """Identify whether a chat call receives end-user content."""

    END_USER = "end_user"
    NON_END_USER = "non_end_user"


class OutputAudience(StrEnum):
    """Identify whether a chat call's response is shown to the end user."""

    END_USER = "end_user"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class LLMRequestPolicy:
    """Describe the input source and output destination of one chat call."""

    input_provenance: InputProvenance
    output_audience: OutputAudience

    @property
    def use_censor(self) -> bool:
        """Whether the gateway must censor this request."""
        return (
            self.input_provenance is InputProvenance.END_USER
            or self.output_audience is OutputAudience.END_USER
        )

    @property
    def content_origin(self) -> str:
        """Return the gateway content-origin classification."""
        if self.output_audience is OutputAudience.END_USER:
            return "FinalPrompt"
        if self.input_provenance is InputProvenance.END_USER:
            return "UserPrompt"
        return "AgentPrompt"

    @property
    def extra_body(self) -> dict[str, bool]:
        """Return the request-body fields the gateway reads.

        Returns:
            dict[str, bool] — Censor flag for the gateway.
        """
        return {"use_censor": self.use_censor}

    @property
    def headers(self) -> dict[str, str]:
        """Return the request headers the gateway reads.

        Returns:
            dict[str, str] — Content origin header for the gateway.
        """
        return {
            CONTENT_ORIGIN_HEADER: self.content_origin,
            SERVICE_NAME_HEADER: (
                os.getenv("SERVICE_NAME", DEFAULT_SERVICE_NAME).strip()
                or DEFAULT_SERVICE_NAME
            ),
        }
