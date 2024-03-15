import pickle
from pathlib import Path

import pytest

from blueetl.adapters import circuit as test_module
from blueetl.adapters.base import AdapterError
from blueetl.adapters.node_sets import NodeSetsAdapter
from tests.unit.utils import BLUEPY_AVAILABLE, TEST_DATA_PATH, assert_isinstance


@pytest.mark.parametrize(
    "path, population, expected_classes",
    [
        (
            pytest.param(
                "sonata/circuit_config.json",
                "default",
                {
                    "circuit": "bluepysnap.Circuit",
                    "population": "bluepysnap.nodes.NodePopulation",
                },
                id="snap",
            )
        ),
        (
            pytest.param(
                "bbp/CircuitConfig",
                None,
                {
                    "circuit": "bluepy.Circuit",
                    "population": "bluepy.cells.CellCollection",
                },
                id="bluepy",
                marks=pytest.mark.skipif(not BLUEPY_AVAILABLE, reason="bluepy not available"),
            )
        ),
    ],
)
def test_circuit_adapter(path, population, expected_classes, monkeypatch):
    path = TEST_DATA_PATH / "circuit" / path
    # enter the circuit dir to resolve relative paths in bluepy
    monkeypatch.chdir(path.parent)
    obj = test_module.CircuitAdapter.from_file(path)
    assert_isinstance(obj.instance, expected_classes["circuit"])

    # access methods and properties
    pop = obj.nodes[population]
    assert_isinstance(pop, expected_classes["population"])

    checksum = obj.checksum()
    assert isinstance(checksum, str)

    # test pickle roundtrip
    dumped = pickle.dumps(obj)
    loaded = pickle.loads(dumped)

    assert isinstance(loaded, test_module.CircuitAdapter)
    assert_isinstance(loaded.instance, expected_classes["circuit"])
    # no cached_properties should be loaded after unpickling
    assert sorted(loaded.__dict__) == ["_impl"]
    assert sorted(loaded._impl.__dict__) == ["_circuit"]


def test_circuit_adapter_with_nonexistent_path():
    path = Path("path/to/circuit_config.json")
    obj = test_module.CircuitAdapter.from_file(path)

    assert obj.instance is None
    assert obj.exists() is False

    with pytest.raises(AdapterError, match="The implementation doesn't exist"):
        _ = obj.nodes

    with pytest.raises(AdapterError, match="The implementation doesn't exist"):
        _ = obj.checksum()

    # test pickle roundtrip
    dumped = pickle.dumps(obj)
    loaded = pickle.loads(dumped)

    assert isinstance(loaded, test_module.CircuitAdapter)
    assert loaded.instance is None


def test_circuit_adapter_node_sets():
    path = TEST_DATA_PATH / "circuit" / "sonata" / "circuit_config.json"
    obj = test_module.CircuitAdapter.from_file(path)

    assert_isinstance(obj.instance, "bluepysnap.Circuit")

    assert isinstance(obj.node_sets_file, Path)
    assert obj.node_sets_file.name == "node_sets.json"

    assert isinstance(obj.node_sets, NodeSetsAdapter)
