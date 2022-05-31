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
    assert isinstance(item, tuple)
    assert item == ("a", "c")
    assert item.i0 == "a"
    assert item.i1 == "c"


def test_iter_without_names(index2):
    obj = index2
    it = obj.etl.iter()
    assert isinstance(it, Iterator)
    item = next(it)
    assert isinstance(item, tuple)
    assert item == ("a", "c")
    assert item.ilevel_0 == "a"
    assert item.ilevel_1 == "c"


def test_iterdict(index1):
    obj = index1
    it = obj.etl.iterdict()
    assert isinstance(it, Iterator)
    item = next(it)
    assert isinstance(item, dict)
    assert item == {"i0": "a", "i1": "c"}


def test_iterdict_without_names(index2):
    obj = index2
    it = obj.etl.iterdict()
    assert isinstance(it, Iterator)
    item = next(it)
    assert isinstance(item, dict)
    assert item == {"ilevel_0": "a", "ilevel_1": "c"}
