from pathlib import Path

import pytest

from blueetl.adapters import node_sets as test_module
from tests.unit.utils import TEST_DATA_PATH, assert_isinstance


def test_node_sets_adapter():
    path = TEST_DATA_PATH / "circuit" / "sonata" / "node_sets.json"
    expected_class_name = "bluepysnap.node_sets.NodeSets"

    obj = test_module.NodeSetsAdapter.from_file(path)

    assert_isinstance(obj.instance, expected_class_name)
    assert obj.exists() is True


def test_node_sets_adapter_update():
    path = TEST_DATA_PATH / "circuit" / "sonata" / "node_sets.json"
    path_extra = TEST_DATA_PATH / "circuit" / "sonata" / "node_sets_extra.json"

    obj = test_module.NodeSetsAdapter.from_file(path)
    obj_extra = test_module.NodeSetsAdapter.from_file(path_extra)

    assert "Layer2" in obj.instance
    assert "ExtraLayer2" not in obj.instance
    assert "ExtraLayer2" in obj_extra.instance

    obj.update(obj_extra)
    assert "ExtraLayer2" in obj.instance


def test_node_sets_adapter_ior():
    path = TEST_DATA_PATH / "circuit" / "sonata" / "node_sets.json"
    path_extra = TEST_DATA_PATH / "circuit" / "sonata" / "node_sets_extra.json"

    obj = test_module.NodeSetsAdapter.from_file(path)
    obj_extra = test_module.NodeSetsAdapter.from_file(path_extra)

    assert "Layer2" in obj.instance
    assert "ExtraLayer2" not in obj.instance
    assert "ExtraLayer2" in obj_extra.instance

    obj |= obj_extra
    assert "ExtraLayer2" in obj.instance


def test_node_sets_adapter_with_none():
    expected_class_name = "bluepysnap.node_sets.NodeSets"
    obj = test_module.NodeSetsAdapter.from_file(None)

    assert_isinstance(obj.instance, expected_class_name)
    assert obj.exists() is True
    assert obj.instance.content == {}


def test_node_sets_adapter_with_nonexistent_path():
    path = Path("path/to/non/existent/file.json")

    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        test_module.NodeSetsAdapter.from_file(path)
