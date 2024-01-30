from pathlib import Path

import numpy as np
import pandas as pd
from bluepysnap import Simulation
from pandas.testing import assert_frame_equal, assert_series_equal

from blueetl.campaign.config import SimulationCampaign
from blueetl.converters import convert_spikes as test_module


def test_main(tmp_path):
    population = "custom"
    input_file = tmp_path / "spikes.csv"
    output_dir = tmp_path / "output"
    input_file.write_text("ids times\n1 10.0\n1 10.1\n2 10.1\n2 20.0\n5 5.0")
    test_module.main(input_file, output_dir, node_population=population)

    assert (output_dir / "analysis_config.yaml").is_file()
    assert (output_dir / "circuit" / "circuit_config.json").is_file()
    assert (output_dir / "circuit" / "node_sets.json").is_file()
    assert (output_dir / "circuit" / "synthetic_nodes.h5").is_file()
    assert (output_dir / "simulation" / "simulation_config.json").is_file()
    assert (output_dir / "simulation" / "synthetic_spikes.h5").is_file()
    assert (output_dir / "simulation_campaign_config.json").is_file()

    campaign = SimulationCampaign.load(output_dir / "simulation_campaign_config.json")
    assert len(campaign) == 1
    simulation = Simulation(campaign[0].path)
    assert (
        Path(simulation._simulation_config_path).resolve()
        == (output_dir / "simulation" / "simulation_config.json").resolve()
    )
    assert (
        Path(simulation.circuit._circuit_config_path).resolve()
        == (output_dir / "circuit" / "circuit_config.json").resolve()
    )

    expected_nodes = pd.DataFrame(
        {"_": np.zeros(6, dtype=np.int8)},
        index=pd.RangeIndex(6, name="node_ids"),
    )
    expected_spikes = pd.Series(
        [5, 1, 1, 2, 2],
        index=pd.Index([5.0, 10.0, 10.1, 10.1, 20.0], name="times"),
        name="ids",
    )
    assert_frame_equal(simulation.circuit.nodes[population].get(), expected_nodes)
    assert_series_equal(simulation.spikes[population].get(), expected_spikes)
