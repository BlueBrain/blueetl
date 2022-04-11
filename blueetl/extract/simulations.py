import hashlib
import json
import logging
from typing import Dict, List

import pandas as pd
from bluepy import Circuit, Simulation
from bluepy.exceptions import BluePyError

from blueetl.config.simulations import SimulationsConfig
from blueetl.constants import (
    CIRCUIT,
    CIRCUIT_ID,
    COMPLETE,
    SIMULATION,
    SIMULATION_ID,
    SIMULATION_PATH,
)
from blueetl.extract.base import BaseExtractor

L = logging.getLogger(__name__)


class Simulations(BaseExtractor):
    COLUMNS = [SIMULATION_PATH, SIMULATION_ID, CIRCUIT_ID, SIMULATION, CIRCUIT, COMPLETE]

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)
        incomplete_simulations = self.df.etl.q(complete=False)
        columns = [SIMULATION_PATH, SIMULATION_ID, CIRCUIT_ID]
        for index, rec in incomplete_simulations[columns].etl.iter():
            L.warning("Ignored simulation without spikes: %s, %s", index, rec)

    @staticmethod
    def _get_circuit_hash(circuit_config: Dict) -> str:
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

    @staticmethod
    def _is_simulation_complete(simulation: Simulation) -> bool:
        """Return True if the spikes can be loaded from the simulation.

        It can be used to filter the simulations before the simulation campaign is complete.
        """
        try:
            _ = simulation.spikes
            return True
        except BluePyError:
            return False

    @classmethod
    def _from_paths(cls, simulation_paths: List[str]) -> pd.DataFrame:
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
            complete = cls._is_simulation_complete(simulation)
            records.append(
                [simulation_id, circuit_id, simulation_path, simulation, circuit, complete]
            )
            L.info(
                "Loading simulation_id=%s, circuit_id=%s, "
                "circuit_hash=%s, simulation_path=%s, complete=%s",
                simulation_id,
                circuit_id,
                circuit_hash,
                simulation_path,
                complete,
            )
        columns = [SIMULATION_ID, CIRCUIT_ID, SIMULATION_PATH, SIMULATION, CIRCUIT, COMPLETE]
        return pd.DataFrame.from_records(records, columns=columns)

    @classmethod
    def from_config(cls, config: SimulationsConfig) -> "Simulations":
        """Load simulations from the given simulation campaign."""
        simulation_paths = config.to_pandas()
        df = cls._from_paths(simulation_paths.to_list())
        # set the conditions
        df.index = simulation_paths.index
        return cls(df)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "Simulations":
        """Load simulations from a dataframe containing valid simulation ids and circuit ids."""
        simulation_paths = df[SIMULATION_PATH]
        new_df = cls._from_paths(simulation_paths.to_list())
        # set the conditions
        new_df.index = simulation_paths.index
        check_columns = [SIMULATION_PATH, SIMULATION_ID, CIRCUIT_ID, COMPLETE]
        difference = new_df[check_columns].compare(df[check_columns])
        if not difference.empty:
            L.error(
                "Inconsistent dataframes, you may need to delete the cache. Differences:\n%s",
                difference,
            )
            raise RuntimeError("Inconsistent dataframes")
        return cls(new_df)

    def to_pandas(self) -> pd.DataFrame:
        """Dump simulations to a dataframe that can be serialized and stored."""
        # skip columns SIMULATION and CIRCUIT because they contain custom runtime objects
        columns = [col for col in self.df.columns if col not in [SIMULATION, CIRCUIT]]
        return self.df[columns]
