import pickle

import bluepy
import bluepysnap
import pytest
import bluepysnap.nodes
from blueetl.adapters import circuit as test_module
from blueetl.adapters.base import AdapterError
from tests.unit.utils import TEST_DATA_PATH
import bluepy.cells


@pytest.mark.parametrize(
    "path, population, expected_classes",
    [
        (
            pytest.param(
                "sonata/circuit_config.json",
                "default",
                {
                    "circuit": bluepysnap.Circuit,
                    "population": bluepysnap.nodes.NodePopulation,
                },
                id="snap",
            )
        ),
        (
            pytest.param(
                "bbp/CircuitConfig",
                None,
                {
                    "circuit": bluepy.Circuit,
                    "population": bluepy.cells.CellCollection,
                },
                id="bluepy",
            )
        ),
    ],
)
def test_circuit_adapter(path, population, expected_classes, monkeypatch):
    path = TEST_DATA_PATH / path
    # enter the circuit dir to resolve relative paths in bluepy
    monkeypatch.chdir(path.parent)
    obj = test_module.CircuitAdapter(TEST_DATA_PATH / path)
    assert isinstance(obj.instance, expected_classes["circuit"])

    # access methods and properties
    pop = obj.nodes[population]
    assert isinstance(pop, expected_classes["population"])

    checksum = obj.checksum()
    assert isinstance(checksum, str)

    # test pickle roundtrip
    dumped = pickle.dumps(obj)
    loaded = pickle.loads(dumped)

    assert isinstance(loaded, test_module.CircuitAdapter)
    assert isinstance(loaded.instance, expected_classes["circuit"])
    # no cached_properties should be loaded after unpickling
    assert sorted(loaded.__dict__) == ["_impl"]
    assert sorted(loaded._impl.__dict__) == ["_circuit"]


def test_circuit_adapter_with_nonexistent_path():
    path = "path/to/circuit_config.json"
    obj = test_module.CircuitAdapter(path)

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
