import pickle
import re

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from blueetl import repository as test_module
from blueetl.extract.neuron_classes import NeuronClasses
from blueetl.extract.neurons import Neurons
from blueetl.extract.simulations import Simulations
from blueetl.extract.spikes import Spikes
from blueetl.extract.windows import Windows
from tests.unit.utils import TEST_DATA_PATH, assert_frame_equal


def test_repository_names(repo):
    assert repo.names == ["simulations", "neurons", "neuron_classes", "windows", "report"]


def test_repository_extract(repo, capsys):
    with pytest.raises(RuntimeError, match="Not all the dataframes have been extracted"):
        repo.check_extractions()
    assert repo.is_extracted() is False
    repo.extract()
    assert repo.is_extracted() is True
    repo.check_extractions()
    repo.show()

    captured = capsys.readouterr()
    pattern = (
        "Extraction: simulations.*"
        "Extraction: neurons.*"
        "Extraction: neuron_classes.*"
        "Extraction: windows.*"
        "Extraction: report.*"
    )
    assert re.search(pattern, captured.out, flags=re.MULTILINE | re.DOTALL)


def test_repository_extract_simulations(repo):
    result = repo.simulations
    assert repo.simulation_ids == [0]
    assert isinstance(result, Simulations)
    expected_columns = [
        "seed",
        "simulation_path",
        "simulation_id",
        "circuit_id",
        "simulation",
        "circuit",
    ]
    assert_array_equal(result.df.columns, expected_columns)
    assert len(result.df) == 1

    missing_simulations = repo.missing_simulations()
    missing_simulation_path = str(
        TEST_DATA_PATH / "simulation_campaign" / "1" / "simulation_config.json"
    )
    expected_df = pd.DataFrame(
        data=[{"seed": 174404, "simulation_path": missing_simulation_path}],
        index=[1],
    )
    assert_frame_equal(missing_simulations, expected_df)


def test_repository_extract_neurons(repo):
    result = repo.neurons
    assert isinstance(result, Neurons)
    expected_columns = ["circuit_id", "neuron_class", "gid", "neuron_class_index"]
    assert_array_equal(result.df.columns, expected_columns)
    assert len(result.df) == 6


def test_repository_extract_neuron_classes(repo):
    result = repo.neuron_classes
    assert isinstance(result, NeuronClasses)
    expected_columns = [
        "circuit_id",
        "neuron_class",
        "count",
        "limit",
        "population",
        "node_set",
        "gids",
        "query",
    ]
    assert_array_equal(result.df.columns, expected_columns)
    assert len(result.df) == 3


def test_repository_extract_windows(repo):
    result = repo.windows
    assert isinstance(result, Windows)
    expected_columns = [
        "simulation_id",
        "circuit_id",
        "window",
        "trial",
        "offset",
        "t_start",
        "t_stop",
        "t_step",
        "duration",
        "window_type",
    ]
    assert_array_equal(result.df.columns, expected_columns)
    assert len(result.df) == 3


def test_repository_extract_report(repo):
    result = repo.report
    assert isinstance(result, Spikes)
    assert repo.spikes is result
    expected_columns = [
        "time",
        "gid",
        "window",
        "trial",
        "simulation_id",
        "circuit_id",
        "neuron_class",
    ]
    assert_array_equal(result.df.columns, expected_columns)
    assert len(result.df) == 16


def test_repository_pickle_roundtrip(repo):
    dumped = pickle.dumps(repo)
    loaded = pickle.loads(dumped)

    assert isinstance(loaded, test_module.Repository)
    assert loaded.extraction_config == repo.extraction_config
    assert loaded.simulations_config == repo.simulations_config
    assert loaded.simulations_filter == repo.simulations_filter
    assert loaded.names == repo.names


def test_repository_apply_filter(repo):
    filtered = repo.apply_filter({})
    assert isinstance(filtered, test_module.FilteredRepository)
