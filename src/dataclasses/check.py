from dataclasses import dataclass


@dataclass
class ComponentCheckResult:
    state: str
    reason: str
