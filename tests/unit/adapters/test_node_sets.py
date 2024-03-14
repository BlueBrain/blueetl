from pathlib import Path

import pytest

from blueetl.adapters import node_sets as test_module
from tests.unit.utils import TEST_NODE_SETS_FILE, TEST_NODE_SETS_FILE_EXTRA, assert_isinstance


def test_node_sets_adapter():
    path = TEST_NODE_SETS_FILE
    expected_class_name = "bluepysnap.node_sets.NodeSets"

    obj = test_module.NodeSetsAdapter.from_file(path)

    assert_isinstance(obj.instance, expected_class_name)
    assert obj.exists() is True


def test_node_sets_adapter_update():
    path = TEST_NODE_SETS_FILE
    path_extra = TEST_NODE_SETS_FILE_EXTRA

    obj = test_module.NodeSetsAdapter.from_file(path)
    obj_extra = test_module.NodeSetsAdapter.from_file(path_extra)

    assert "Layer2" in obj.instance
    assert "ExtraLayer2" not in obj.instance
    assert "ExtraLayer2" in obj_extra.instance

    obj.update(obj_extra)
    assert "ExtraLayer2" in obj.instance


def test_node_sets_adapter_ior():
    path = TEST_NODE_SETS_FILE
    path_extra = TEST_NODE_SETS_FILE_EXTRA

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
