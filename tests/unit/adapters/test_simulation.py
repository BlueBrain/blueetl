import pickle
from pathlib import Path

import pytest

from blueetl.adapters import simulation as test_module
from blueetl.adapters.base import AdapterError
from tests.unit.utils import BLUEPY_AVAILABLE, TEST_DATA_PATH, assert_isinstance


@pytest.mark.parametrize(
    "path, population, reports, expected_classes",
    [
        (
            pytest.param(
                "sonata/simulation_config.json",
                "default",
                ["soma_report", "section_report"],
                {
                    "simulation": "bluepysnap.Simulation",
                    "population": "bluepysnap.nodes.NodePopulation",
                    "spikes": "bluepysnap.spike_report.PopulationSpikeReport",
                    "soma_report": "bluepysnap.frame_report.PopulationSomaReport",
                    "section_report": "bluepysnap.frame_report.PopulationCompartmentReport",
                },
                id="snap",
            )
        ),
        (
            pytest.param(
                "bbp/BlueConfig",
                None,
                ["soma", "AllCompartments"],
                {
                    "simulation": "bluepy.Simulation",
                    "population": "bluepy.cells.CellCollection",
                    "spikes": "blueetl.adapters.bluepy.simulation.PopulationSpikesReportImpl",
                    "soma": "blueetl.adapters.bluepy.simulation.PopulationReportImpl",
                    "AllCompartments": "blueetl.adapters.bluepy.simulation.PopulationReportImpl",
                },
                id="bluepy",
                marks=pytest.mark.skipif(not BLUEPY_AVAILABLE, reason="bluepy not available"),
            )
        ),
    ],
)
def test_simulation_adapter(path, population, reports, expected_classes, monkeypatch):
    path = TEST_DATA_PATH / "simulation" / path
    # enter the circuit dir to resolve relative paths in bluepy
    monkeypatch.chdir(path.parent)
    obj = test_module.SimulationAdapter.from_file(path)
    assert_isinstance(obj.instance, expected_classes["simulation"])

    assert obj.exists() is True
    assert obj.is_complete() is True

    # access methods and properties
    pop = obj.circuit.nodes[population]
    assert_isinstance(pop, expected_classes["population"])

    spikes = obj.spikes[population]
    assert_isinstance(spikes, expected_classes["spikes"])

    for report_name in reports:
        report = obj.reports[report_name][population]
        assert_isinstance(report, expected_classes[report_name])

    # test pickle roundtrip
    dumped = pickle.dumps(obj)
    loaded = pickle.loads(dumped)

    assert isinstance(loaded, test_module.SimulationAdapter)
    assert_isinstance(loaded.instance, expected_classes["simulation"])
    # no cached_properties should be loaded after unpickling
    assert sorted(loaded.__dict__) == ["_impl"]
    assert sorted(loaded._impl.__dict__) == ["_simulation"]


def test_simulation_adapter_with_nonexistent_path():
    path = Path("path/to/simulation_config.json")
    obj = test_module.SimulationAdapter.from_file(path)

    assert obj.instance is None
    assert obj.exists() is False
    assert obj.is_complete() is False

    with pytest.raises(AdapterError, match="The implementation doesn't exist"):
        _ = obj.circuit

    with pytest.raises(AdapterError, match="The implementation doesn't exist"):
        _ = obj.spikes

    with pytest.raises(AdapterError, match="The implementation doesn't exist"):
        _ = obj.reports

    # test pickle roundtrip
    dumped = pickle.dumps(obj)
    loaded = pickle.loads(dumped)

    assert isinstance(loaded, test_module.SimulationAdapter)
    assert loaded.instance is None
