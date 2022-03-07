import hashlib
import json
import logging
from typing import Dict, List

import pandas as pd
from bluepy import Circuit, Simulation

from blueetl.constants import CIRCUIT, CIRCUIT_ID, SIMULATION, SIMULATION_ID, SIMULATION_PATH
from blueetl.utils import ensure_dtypes

L = logging.getLogger(__name__)


class Simulations:
    def __init__(self, df: pd.DataFrame):
        """Initialize a Simulations object.

        Args:
            df: DataFrame with
                columns: simulation_path, simulation_id, circuit_id, simulation, circuit
                index: simulation coordinates
        """
        assert set(df.columns) == {SIMULATION_PATH, SIMULATION_ID, CIRCUIT_ID, SIMULATION, CIRCUIT}
        self._df = ensure_dtypes(df)

    @property
    def df(self):
        return self._df

    @staticmethod
    def _get_circuit_hash(circuit_config):
        # TODO: verify which keys to consider, or use the circuit path?
        circuit_config_keys = [
            "cells",
            "morphologies",
            # "morphology_type",
            "emodels",
            # "mecombo_info",
            "connectome",
            # "targets",
            "projections",
            # "projections_metadata",
            # "segment_index",
            # "synapse_index",
            "atlas",
        ]
        s = json.dumps({k: circuit_config.get(k) for k in circuit_config_keys}, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    @classmethod
    def from_paths(cls, simulation_paths: List[str]):
        """Return a dataframe of simulations from a list of simulation paths."""
        circuit_hashes: Dict[str, int] = {}  # map circuit_hash -> circuit_id
        circuits: Dict[int, Circuit] = {}  # map circuit_id -> circuit
        records = []
        for simulation_id, simulation_path in enumerate(simulation_paths):
            simulation = Simulation(simulation_path)
            circuit_hash = cls._get_circuit_hash(simulation.circuit.config)
            # if circuit_hash is new, use simulation_id as circuit_id
            circuit_id = circuit_hashes.setdefault(circuit_hash, simulation_id)
            circuit = circuits.setdefault(circuit_id, simulation.circuit)
            records.append([simulation_id, circuit_id, simulation_path, simulation, circuit])
            L.info(
                "Loading simulation_id=%s, circuit_id=%s, circuit_hash=%s, simulation_path=%s",
                simulation_id,
                circuit_id,
                circuit_hash,
                simulation_path,
            )
        columns = [SIMULATION_ID, CIRCUIT_ID, SIMULATION_PATH, SIMULATION, CIRCUIT]
        return pd.DataFrame.from_records(records, columns=columns)

    @classmethod
    def from_config(cls, config):
        """Load simulations from the given simulation campaign."""
        simulation_paths = config.to_pandas()
        df = cls.from_paths(list(simulation_paths))
        df.index = simulation_paths.index
        return cls(df)

    @classmethod
    def from_pandas(cls, df):
        """Load simulations from a dataframe containing valid simulation ids and circuit ids."""
        simulation_paths = df[SIMULATION_PATH]
        new_df = cls.from_paths(list(simulation_paths))
        new_df.index = df.index
        check_columns = [SIMULATION_PATH, SIMULATION_ID, CIRCUIT_ID]
        difference = new_df[check_columns].compare(df[check_columns])
        if not difference.empty:
            raise ValueError(f"Invalid ids in the input dataframe. Difference:\n{difference}")
        return cls(new_df)

    def to_pandas(self):
        """Dump simulations to a dataframe that can be serialized and stored."""
        # skip columns SIMULATION, CIRCUIT because they contain custom runtime objects
        return self.df[[SIMULATION_ID, CIRCUIT_ID, SIMULATION_PATH]]
