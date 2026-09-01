"""Minimal registry: maps method names to constructors/experiment runners."""

_METHODS = {}


def register(name):
    def deco(fn):
        _METHODS[name] = fn
        return fn
    return deco


def get_method(name):
    return _METHODS[name]


def available():
    return sorted(_METHODS)
