"""A small named-component registry used for dependency injection.

Executors, environment collectors and aggregators register under a name;
configuration selects them by name, and tests inject fakes the same way.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar, overload

T = TypeVar("T")


class UnknownComponentError(KeyError):
    """Raised when configuration names a component that is not registered."""


class ComponentRegistry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    @overload
    def register(self, name: str) -> Callable[[T], T]: ...
    @overload
    def register(self, name: str, item: T) -> T: ...
    def register(self, name: str, item: T | None = None) -> T | Callable[[T], T]:
        def _add(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._items[name] = obj
            return obj

        return _add if item is None else _add(item)

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "none"
            raise UnknownComponentError(
                f"unknown {self.kind} {name!r}; available: {available}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)
