from typing import Iterator

from blueetl.core.etl import ETLIndexAccessor


def test_etl_instance(index1):
    obj = index1
    result = obj.etl
    assert isinstance(result, ETLIndexAccessor)


def test_iter(index1):
    obj = index1
    it = obj.etl.iter()
    assert isinstance(it, Iterator)
    item = next(it)
    assert item == ("a", "c")
    assert item.i0 == "a"
    assert item.i1 == "c"
