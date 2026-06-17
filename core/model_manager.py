from typing import Any

_store: dict[str, Any] = {}


def get(name: str) -> Any:
    return _store.get(name)


def set(name: str, obj: Any) -> None:
    _store[name] = obj


def loaded(name: str) -> bool:
    return name in _store


def clear(name: str) -> None:
    _store.pop(name, None)
