from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from src.schemas.usage import (
    emitted_usage,
    extract_usage,
    max_usage,
    merge_usage,
    zero_usage,
)


class UsageCollector(AsyncCallbackHandler):
    """Accumulate per-request usage sums and single-call peaks."""

    def __init__(self) -> None:
        self._total = zero_usage()
        self._peak = zero_usage()
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def total(self) -> dict[str, int]:
        """Return the unwrapped per-request usage sum.

        Returns:
            dict[str, int] — Copy of the accumulated usage totals.
        """
        return dict(self._total)

    @property
    def peak(self) -> dict[str, int]:
        return dict(self._peak)

    def record(self, usage: dict[str, int]) -> None:
        """Fold one call's usage into both channels.

        Args:
            usage: dict[str, int] — Usage values from one model call.
        """
        if not any(usage.values()):
            return
        self._total = merge_usage(self._total, usage)
        self._peak = max_usage(self._peak, usage)
        self._calls += 1

    def payload(self) -> dict[str, int]:
        """Build the wire usage payload for this request.

        Returns:
            dict[str, int] — Total and peak usage values.
        """
        return emitted_usage(self._total, self._peak)

    async def on_llm_end(
        self, response: LLMResult, *, run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        usage = extract_usage(response.llm_output)
        for generations in response.generations:
            for generation in generations:
                usage = extract_usage(getattr(generation, "message", generation))
                if any(usage.values()):
                    break
            if any(usage.values()):
                break
        self.record(usage)
