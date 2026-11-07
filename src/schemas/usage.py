from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "total_tokens",
)

_USAGE_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "input_tokens": ("input_tokens", "prompt_tokens"),
    "output_tokens": ("output_tokens", "completion_tokens"),
    "cached_input_tokens": (
        "cached_input_tokens",
        "cached_tokens",
        "cache_read",
        "cache_read_tokens",
    ),
    "reasoning_tokens": ("reasoning_tokens", "reasoning"),
}

#: Containers a provider may bury the usage block inside.
_NESTED_USAGE_KEYS = (
    "message",
    "usage_metadata",
    "usage",
    "response_metadata",
    "generation_info",
    "token_usage",
    "input_token_details",
    "output_token_details",
    "prompt_tokens_details",
    "completion_tokens_details",
)


def zero_usage() -> dict[str, int]:
    return {key: 0 for key in _USAGE_KEYS}


def _with_total(usage: dict[str, int]) -> dict[str, int]:
    """Derive total tokens from input and output tokens.

    Args:
        usage: dict[str, int] — Usage values to augment.

    Returns:
        dict[str, int] — Usage values with a derived total.
    """
    usage["total_tokens"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return usage


def normalize_usage(usage: Any | None) -> dict[str, int]:
    """Normalize a provider usage object into the canonical dictionary.

    Args:
        usage: Any | None — Provider usage object to normalize.

    Returns:
        dict[str, int] — Canonical usage values.
    """
    if usage is None:
        return zero_usage()

    candidates = _iter_usage_candidates(usage)
    normalized = zero_usage()
    for field, aliases in _USAGE_KEY_ALIASES.items():
        value = _first_int_across(candidates, aliases)
        if value is not None:
            normalized[field] = value
    return _with_total(normalized)


def extract_usage(message_or_chunk: Any) -> dict[str, int]:
    """Extract usage from a LangChain message, chunk, or result.

    Args:
        message_or_chunk: Any — Object that may contain usage metadata.

    Returns:
        dict[str, int] — Extracted canonical usage values.
    """
    if message_or_chunk is None:
        return zero_usage()

    usage = normalize_usage(message_or_chunk)
    if any(usage.values()):
        return usage

    for attribute in ("message", "additional_kwargs"):
        nested = getattr(message_or_chunk, attribute, None)
        if nested is not None and nested is not message_or_chunk:
            usage = normalize_usage(nested)
            if any(usage.values()):
                return usage

    return zero_usage()


def merge_usage(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """Sum two usage payloads.

    Args:
        a: dict[str, int] — First usage payload.
        b: dict[str, int] — Second usage payload.

    Returns:
        dict[str, int] — Per-request usage total.
    """
    merged = zero_usage()
    for usage in (a, b):
        for key, value in normalize_usage(usage).items():
            merged[key] += value
    return merged


def _carried_total(usage: dict[str, int]) -> int:
    """Return a payload's carried or derived single-call total.

    Args:
        usage: dict[str, int] — Usage payload to inspect.

    Returns:
        int — Carried total tokens or derived input-plus-output total.
    """
    if isinstance(usage, Mapping):
        total = usage.get("total_tokens")
        if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
            return total
    normalized = normalize_usage(usage)
    return normalized["input_tokens"] + normalized["output_tokens"]


def max_usage(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """Keep the per-field maximum from two usage payloads.

    Args:
        a: dict[str, int] — First usage payload.
        b: dict[str, int] — Second usage payload.

    Returns:
        dict[str, int] — Per-field maxima with the largest single-call total.
    """
    merged = zero_usage()
    na, nb = normalize_usage(a), normalize_usage(b)
    for key in merged:
        merged[key] = max(na[key], nb[key])
    merged["total_tokens"] = max(_carried_total(a), _carried_total(b))
    return merged


def emitted_usage(total: dict[str, int], peak: dict[str, int]) -> dict[str, int]:
    """Assemble wire usage totals and per-call peaks.

    Args:
        total: dict[str, int] — Per-request usage totals.
        peak: dict[str, int] — Largest single-call usage values.

    Returns:
        dict[str, int] — Wire usage payload.
    """
    return {
        **total,
        "peak_input_tokens": peak.get("input_tokens", 0),
        "peak_output_tokens": peak.get("output_tokens", 0),
        "peak_reasoning_tokens": peak.get("reasoning_tokens", 0),
        "peak_total_tokens": peak.get("total_tokens", 0),
    }


def _iter_usage_candidates(usage: Any) -> list[Mapping[str, Any]]:
    """Walk containers that may hold a usage block breadth-first.

    Args:
        usage: Any — Provider-shaped object to traverse.

    Returns:
        list[Mapping[str, Any]] — Mapping candidates that may contain usage.
    """
    queue: list[Any] = [usage]
    seen: set[int] = set()
    candidates: list[Mapping[str, Any]] = []

    while queue:
        mapping = _as_mapping(queue.pop(0))
        if mapping is None or id(mapping) in seen:
            continue
        seen.add(id(mapping))
        candidates.append(mapping)

        for key in _NESTED_USAGE_KEYS:
            nested = mapping.get(key)
            if nested is not None:
                queue.append(nested)

    return candidates


def _first_int_across(
    candidates: list[Mapping[str, Any]], aliases: tuple[str, ...]
) -> int | None:
    for candidate in candidates:
        for alias in aliases:
            value = candidate.get(alias)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
        except TypeError:
            dumped = None
        if isinstance(dumped, Mapping):
            return dumped

    data = getattr(value, "__dict__", None)
    return data if isinstance(data, dict) else None
