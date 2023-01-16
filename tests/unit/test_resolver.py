from unittest.mock import Mock

import pytest

from blueetl import resolver as test_module


def test_object_resolver():
    obj = Mock()
    obj.a.b.c = 123
    resolver = test_module.ObjectResolver(obj)

    result = resolver.get("a.b.c")

    assert result == 123


def test_object_resolver_with_level():
    obj = Mock()
    obj.a.b.c = 123
    resolver = test_module.ObjectResolver(obj)

    result = resolver.get("a.b.c", level=None)
    assert result == 123

    result = resolver.get("a.b.c", level=0)
    assert result is obj

    result = resolver.get("a.b.c", level=1)
    assert result is obj.a

    result = resolver.get("a.b.c", level=2)
    assert result is obj.a.b

    result = resolver.get("a.b.c", level=3)
    assert result is obj.a.b.c

    result = resolver.get("a.b.c", level=4)
    assert result is obj.a.b.c

    result = resolver.get("a.b.c", level=-1)
    assert result is obj.a.b

    result = resolver.get("a.b.c", level=-2)
    assert result is obj.a

    result = resolver.get("a.b.c", level=-3)
    assert result is obj

    result = resolver.get("a.b.c", level=-4)
    assert result is obj


def test_object_resolver_error():
    obj = Mock()
    obj.a.b = 123
    resolver = test_module.ObjectResolver(obj)

    with pytest.raises(test_module.ResolverError, match="Impossible to resolve a.b.c at level c"):
        resolver.get("a.b.c")


def test_dict_resolver():
    obj = {"a": {"b": {"c": 123}}}
    resolver = test_module.DictResolver(obj)

    result = resolver.get("a.b.c")

    assert result == 123


def test_dict_resolver_error():
    obj = {"a": {"b": 123}}
    resolver = test_module.DictResolver(obj)

    with pytest.raises(test_module.ResolverError, match="Impossible to resolve a.b.c at level c"):
        resolver.get("a.b.c")


def test_dict_resolver_with_level():
    obj = {"a": {"b": {"c": 123}}}
    resolver = test_module.DictResolver(obj)

    result = resolver.get("a.b.c", level=None)
    assert result == 123

    result = resolver.get("a.b.c", level=0)
    assert result is obj

    result = resolver.get("a.b.c", level=1)
    assert result is obj["a"]

    result = resolver.get("a.b.c", level=2)
    assert result is obj["a"]["b"]

    result = resolver.get("a.b.c", level=3)
    assert result is obj["a"]["b"]["c"]

    result = resolver.get("a.b.c", level=4)
    assert result is obj["a"]["b"]["c"]

    result = resolver.get("a.b.c", level=-1)
    assert result is obj["a"]["b"]

    result = resolver.get("a.b.c", level=-2)
    assert result is obj["a"]

    result = resolver.get("a.b.c", level=-3)
    assert result is obj

    result = resolver.get("a.b.c", level=-4)
    assert result is obj
