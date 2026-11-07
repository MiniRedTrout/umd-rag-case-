import asyncio
from typing import Awaitable, Callable, Protocol, Set

from src.dataclasses.check import ComponentCheckResult


class SupportsCheck(Protocol):
    name: str

    async def check(self) -> bool:
        ...


class HealthCheckService:
    def __init__(self):
        self.__registered_components: Set[SupportsCheck] = set()

    def register_component(self, component: SupportsCheck):
        if component not in self.__registered_components:
            self.__registered_components.add(component)

    async def check(self) -> ComponentCheckResult:
        try:
            results = await asyncio.gather(
                *[component.check() for component in self.__registered_components], return_exceptions=True
            )
            for res in results:
                if isinstance(res, Exception) or res is False:
                    return ComponentCheckResult(
                        state="FAILED", reason=str(res) if isinstance(res, Exception) else "Component check failed"
                    )
            return ComponentCheckResult(state="ACTIVE", reason="everything is OK")

        except Exception as e:
            return ComponentCheckResult(state="UNREACHABLE", reason=str(e))


class CheckableComponentWrapper(SupportsCheck):
    def __init__(self, name: str, func: Callable[..., Awaitable[bool]], *args, **kwargs):
        self.name = name
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return getattr(other, "name", None) == self.name

    async def check(self) -> bool:
        return await self.func(*self.args, **self.kwargs)

    @classmethod
    def build_from_func(cls, name: str, func: Callable[..., Awaitable[bool]], *args, **kwargs):
        return cls(name, func, *args, **kwargs)
