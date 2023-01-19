from unittest.mock import Mock

import pytest

from blueetl import resolver as test_module


def test_object_resolver():
    obj = Mock()
    obj.a.b.c = 123
    resolver = test_module.AttrResolver(obj)

    result = resolver.get("a.b.c")

    assert result == 123


def test_object_resolver_error():
    obj = Mock()
    obj.a.b = 123
    resolver = test_module.AttrResolver(obj)

    with pytest.raises(test_module.ResolverError, match="Impossible to resolve a.b.c at level c"):
        resolver.get("a.b.c")


def test_dict_resolver():
    obj = {"a": {"b": {"c": 123}}}
    resolver = test_module.ItemResolver(obj)

    result = resolver.get("a.b.c")

    assert result == 123


def test_dict_resolver_error():
    obj = {"a": {"b": 123}}
    resolver = test_module.ItemResolver(obj)

    with pytest.raises(test_module.ResolverError, match="Impossible to resolve a.b.c at level c"):
        resolver.get("a.b.c")
