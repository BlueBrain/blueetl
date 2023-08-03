import pickle

import bluepy.cells
import bluepy.impl.compartment_report
import bluepy.impl.spike_report
import bluepysnap.frame_report
import bluepysnap.nodes
import bluepysnap.spike_report
import pytest

from blueetl.adapters import simulation as test_module
from blueetl.adapters.base import AdapterError
from blueetl.adapters.bluepy.simulation import PopulationReportImpl, PopulationSpikesReportImpl
from tests.unit.utils import TEST_DATA_PATH


@pytest.mark.parametrize(
    "path, population, reports, expected_classes",
    [
        (
            pytest.param(
                "sonata/simulation_config.json",
                "default",
                ["soma_report", "section_report"],
                {
                    "simulation": bluepysnap.Simulation,
                    "population": bluepysnap.nodes.NodePopulation,
                    "spikes": bluepysnap.spike_report.PopulationSpikeReport,
                    "soma_report": bluepysnap.frame_report.PopulationSomaReport,
                    "section_report": bluepysnap.frame_report.PopulationCompartmentReport,
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
                    "simulation": bluepy.Simulation,
                    "population": bluepy.cells.CellCollection,
                    "spikes": PopulationSpikesReportImpl,
                    "soma": PopulationReportImpl,
                    "AllCompartments": PopulationReportImpl,
                },
                id="bluepy",
            )
        ),
    ],
)
def test_simulation_adapter(path, population, reports, expected_classes, monkeypatch):
    path = TEST_DATA_PATH / path
    # enter the circuit dir to resolve relative paths in bluepy
    monkeypatch.chdir(path.parent)
    obj = test_module.SimulationAdapter(TEST_DATA_PATH / path)
    assert isinstance(obj.instance, expected_classes["simulation"])

    assert obj.exists() is True
    assert obj.is_complete() is True

    # access methods and properties
    pop = obj.circuit.nodes[population]
    assert isinstance(pop, expected_classes["population"])

    spikes = obj.spikes[population]
    assert isinstance(spikes, expected_classes["spikes"])

    for report_name in reports:
        report = obj.reports[report_name][population]
        assert isinstance(report, expected_classes[report_name])

    # test pickle roundtrip
    dumped = pickle.dumps(obj)
    loaded = pickle.loads(dumped)

    assert isinstance(loaded, test_module.SimulationAdapter)
    assert isinstance(loaded.instance, expected_classes["simulation"])
    # no cached_properties should be loaded after unpickling
    assert sorted(loaded.__dict__) == ["_impl"]
    assert sorted(loaded._impl.__dict__) == ["_simulation"]


def test_simulation_adapter_with_nonexistent_path():
    path = "path/to/simulation_config.json"
    obj = test_module.SimulationAdapter(path)

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
