from __future__ import annotations

import pytest

from qcal.components import ComponentRegistry, UnknownComponentError


def test_register_by_decorator_and_directly() -> None:
    registry: ComponentRegistry[object] = ComponentRegistry("widget")

    @registry.register("a")
    def a() -> str:
        return "a"

    registry.register("b", len)
    assert registry.get("a") is a
    assert registry.get("b") is len
    assert registry.names() == ["a", "b"]
    assert list(registry) == ["a", "b"]
    assert len(registry) == 2
    assert "a" in registry
    assert "z" not in registry


def test_duplicate_names_are_rejected() -> None:
    registry: ComponentRegistry[int] = ComponentRegistry("number")
    registry.register("x", 1)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("x", 2)


def test_unknown_names_list_what_is_available() -> None:
    registry: ComponentRegistry[int] = ComponentRegistry("number")
    with pytest.raises(UnknownComponentError, match="available: none"):
        registry.get("x")
    registry.register("one", 1)
    with pytest.raises(UnknownComponentError, match="available: one"):
        registry.get("x")
