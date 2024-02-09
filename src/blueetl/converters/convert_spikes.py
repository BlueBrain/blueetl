"""Convert a spikes file by generating synthetic circuit, simulation, and simulation campaign."""

import dataclasses
import logging
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from blueetl.utils import dump_json, dump_yaml, import_optional_dependency, relpath, resolve_path

L = logging.getLogger(__name__)

TIMESTAMPS = "timestamps"
NODE_IDS = "node_ids"
DTYPES = {
    TIMESTAMPS: np.float64,
    NODE_IDS: np.uint64,
}


@dataclasses.dataclass
class DataStats:
    """Statistics on the imported spikes."""

    rows: int
    unique_ids: int
    min_id: int
    max_id: int
    min_timestamp: float
    max_timestamp: float


@dataclasses.dataclass
class OutputPaths:
    """Output paths."""

    base: Path

    @property
    def circuit_path(self) -> Path:
        """Return the circuit_path."""
        return self.base / "circuit"

    @property
    def simulation_path(self) -> Path:
        """Return the simulation_path."""
        return self.base / "simulation"

    @property
    def spikes_path(self) -> Path:
        """Return the spikes_path."""
        return self.simulation_path / "synthetic_spikes.h5"

    @property
    def nodes_path(self) -> Path:
        """Return the nodes_path."""
        return self.circuit_path / "synthetic_nodes.h5"

    @property
    def node_sets_path(self) -> Path:
        """Return the node_sets_path."""
        return self.circuit_path / "node_sets.json"

    @property
    def circuit_config_path(self) -> Path:
        """Return the circuit_config_path."""
        return self.circuit_path / "circuit_config.json"

    @property
    def simulation_config_path(self) -> Path:
        """Return the simulation_config_path."""
        return self.simulation_path / "simulation_config.json"

    @property
    def simulation_campaign_config_path(self) -> Path:
        """Return the simulation_campaign_config_path."""
        return self.base / "simulation_campaign_config.json"

    @property
    def analysis_config_path(self) -> Path:
        """Return the analysis_config_path."""
        return self.base / "analysis_config.yaml"

    def mkdirs(self):
        """Create the directories."""
        self.base.mkdir(exist_ok=True, parents=True)
        self.circuit_path.mkdir(exist_ok=True, parents=True)
        self.simulation_path.mkdir(exist_ok=True, parents=True)


def _load_csv(path: Path, sep: str = " ", **kwargs) -> pd.DataFrame:
    """Load and sort spikes data from csv file.

    Accepted column names: timestamps or times, node_ids or ids.
    """
    columns = [TIMESTAMPS, NODE_IDS]
    valid_columns = {*columns, "ids", "times"}
    df = pd.read_csv(path, sep=sep, usecols=lambda x: x in valid_columns, **kwargs)
    df = df.rename(columns={"ids": NODE_IDS, "times": TIMESTAMPS})
    if missing := set(columns).difference(df.columns):
        raise ValueError(f"Missing columns in the CSV file: {missing}")
    df = df[columns].sort_values(columns)
    L.info("Loaded file %s", path)
    return df


def _get_data_stats(df: pd.DataFrame) -> DataStats:
    """Calculate statistics on the imported spikes."""
    unique_ids = df[NODE_IDS].drop_duplicates()
    stats = DataStats(
        rows=len(df),
        unique_ids=len(unique_ids),
        min_id=unique_ids.min(),
        max_id=unique_ids.max(),
        min_timestamp=df[TIMESTAMPS].min(),
        max_timestamp=df[TIMESTAMPS].max(),
    )
    L.info(
        "CSV stats: rows=%s, unique_ids=%s, min_id=%s, max_id=%s, min_ts=%s, max_ts=%s",
        stats.rows,
        stats.unique_ids,
        stats.min_id,
        stats.max_id,
        stats.min_timestamp,
        stats.max_timestamp,
    )
    return stats


def _write_spikes(
    path: Path, node_population: str, timestamps: np.ndarray, node_ids: np.ndarray
) -> None:
    """Write a spikes file in SONATA format."""
    L.info("Writing %s", path)
    sorting_type = h5py.enum_dtype({"none": 0, "by_id": 1, "by_time": 2})
    with h5py.File(path, "w") as h5:
        root = h5.create_group("spikes")
        pop = root.create_group(node_population)
        pop.attrs.create("sorting", data=2, dtype=sorting_type)
        ts_dataset = pop.create_dataset(TIMESTAMPS, data=timestamps, dtype=DTYPES[TIMESTAMPS])
        ts_dataset.attrs.create("units", "ms")
        pop.create_dataset(NODE_IDS, data=node_ids, dtype=DTYPES[NODE_IDS])


def _write_circuit(path: Path, node_population: str, size: int) -> None:
    """Write a synthetic empty circuit in SONATA format."""
    L.info("Writing %s", path)
    voxcell = import_optional_dependency("voxcell")
    # CellCollection uses 1-based ids, since it predates SONATA
    nodes = pd.DataFrame(index=range(1, size + 1))
    nodes["_"] = np.zeros(size, dtype=np.int8)
    cc = voxcell.CellCollection.from_dataframe(nodes)
    cc.population_name = node_population
    cc.save(str(path))


def _write_node_sets(path: Path, node_population: str) -> None:
    """Write a partial node_sets_file."""
    L.info("Writing %s", path)
    node_sets = {
        "empty": {
            "population": node_population,
            "node_id": [],
        }
    }
    dump_json(path, node_sets)


def _write_circuit_config(
    path: Path, node_sets_path: Path, nodes_path: Path, node_population: str
) -> None:
    """Write a partial circuit config in SONATA format."""
    L.info("Writing %s", path)
    nodes_file = str(relpath(nodes_path, path.parent))
    node_sets_file = str(relpath(node_sets_path, path.parent))
    circuit_config = {
        "version": "2.4",
        "metadata": {"status": "partial"},
        "node_sets_file": node_sets_file,
        "networks": {
            "nodes": [
                {
                    "nodes_file": nodes_file,
                    "populations": {node_population: {}},
                }
            ],
            "edges": [],
        },
    }
    dump_json(path, circuit_config)


def _write_simulation_config(
    path: Path, circuit_config_path: Path, spikes_path: Path, tstop: float
) -> None:
    """Write a simulation config in SONATA format."""
    L.info("Writing %s", path)
    network = str(relpath(circuit_config_path, path.parent))
    output_dir = str(relpath(spikes_path.parent, path.parent))
    spikes_file = spikes_path.name
    simulation_config = {
        "version": "2.4",
        "network": network,
        "run": {
            "tstop": tstop,
            "dt": 1.0,
            "random_seed": 0,
        },
        "output": {
            "output_dir": output_dir,
            "spikes_file": spikes_file,
        },
        "metadata": {
            "note": "Synthetic simulation",
        },
    }
    dump_json(path, simulation_config)


def _write_simulation_campaign_config(
    path: Path, circuit_config_path: Path, simulation_config_path: Path
) -> None:
    """Write a simulation campaign config in BlueETL format, having only one simulation."""
    L.info("Writing %s", path)
    circuit_config = str(resolve_path(circuit_config_path))
    path_prefix = str(resolve_path(simulation_config_path.parents[1]))
    simulation_path = str(relpath(simulation_config_path, path_prefix))
    simulation_campaign_config = {
        "format": "blueetl",
        "version": 1,
        "name": "synthetic",
        "attrs": {
            "path_prefix": path_prefix,
            "circuit_config": circuit_config,
            "__coupled__": "coupled",
        },
        "data": [
            {"simulation_path": simulation_path},
        ],
    }
    dump_yaml(path, simulation_campaign_config)


def _write_analysis_config(path: Path, simulation_campaign_config_path: Path) -> None:
    """Write an analysis config in BlueETL format."""
    L.info("Writing %s", path)
    simulation_campaign = str(relpath(simulation_campaign_config_path, path.parent))
    analysis_config = {
        "version": 3,
        "simulation_campaign": simulation_campaign,
        "output": "analysis",
        "analysis": {
            "spikes": {
                "extraction": {
                    "report": {"type": "spikes"},
                    "neuron_classes": {
                        "all": {},
                    },
                    "limit": None,
                    "population": "synthetic",
                    "node_set": None,
                    "windows": {
                        "w1": {"bounds": [0.0, 1000.0]},
                        "w2": {"bounds": [1000.0, 2000.0]},
                    },
                },
                "features": [
                    {
                        "type": "multi",
                        "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                        "function": (
                            "blueetl.external.bnac.calculate_features.calculate_features_multi"
                        ),
                        "params": {"export_all_neurons": True},
                    }
                ],
            }
        },
    }
    dump_yaml(path, analysis_config)


def main(input_file: Path, output_dir: Path, node_population: str) -> None:
    """Read a CSV file containing the spikes, and write synthetic files to be used with BlueETL.

    Expected output files:

    ├── analysis_config.yaml
    ├── circuit
    │   ├── circuit_config.json
    │   ├── node_sets.json
    │   └── synthetic_nodes.h5
    ├── simulation
    │   ├── simulation_config.json
    │   └── synthetic_spikes.h5
    └── simulation_campaign_config.json

    """
    paths = OutputPaths(base=output_dir)
    paths.mkdirs()
    df = _load_csv(input_file)
    stats = _get_data_stats(df)

    _write_circuit(
        paths.nodes_path,
        node_population=node_population,
        size=stats.max_id + 1,
    )
    _write_node_sets(
        paths.node_sets_path,
        node_population=node_population,
    )
    _write_circuit_config(
        paths.circuit_config_path,
        nodes_path=paths.nodes_path,
        node_sets_path=paths.node_sets_path,
        node_population=node_population,
    )
    _write_spikes(
        paths.spikes_path,
        node_population=node_population,
        timestamps=df[TIMESTAMPS],
        node_ids=df[NODE_IDS],
    )
    _write_simulation_config(
        paths.simulation_config_path,
        circuit_config_path=paths.circuit_config_path,
        spikes_path=paths.spikes_path,
        tstop=float(math.ceil(stats.max_timestamp)),
    )
    _write_simulation_campaign_config(
        paths.simulation_campaign_config_path,
        circuit_config_path=paths.circuit_config_path,
        simulation_config_path=paths.simulation_config_path,
    )
    _write_analysis_config(
        paths.analysis_config_path,
        simulation_campaign_config_path=paths.simulation_campaign_config_path,
    )
