import hashlib
import json
import logging
from typing import Dict, List

import pandas as pd
from bluepy import Circuit, Simulation
from bluepy.exceptions import BluePyError

from blueetl.config.simulations import SimulationsConfig
from blueetl.constants import CIRCUIT, CIRCUIT_ID, SIMULATION, SIMULATION_ID, SIMULATION_PATH
from blueetl.extract.base import BaseExtractor

L = logging.getLogger(__name__)


class Simulations(BaseExtractor):
    COLUMNS = [SIMULATION_PATH, SIMULATION_ID, CIRCUIT_ID, SIMULATION, CIRCUIT]

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

        It can be used to ignore a simulation before the simulation campaign is complete.
        """
        try:
            _ = simulation.spikes
            return True
        except BluePyError:
            return False

    @classmethod
    def _from_paths(cls, simulation_paths: pd.Series, load_spikes: bool = True) -> pd.DataFrame:
        """Return a dataframe of simulations from a list of simulation paths.

        Args:
            simulation_paths: Series of paths.
            load_spikes: if True, load the spikes to verify that they are available,
                and add the simulation only in that case.
                If False, don't load the spikes and always add the simulation.

        Returns:
            DataFrame of simulations, using the same index of simulation_paths.
            Simulations without spikes are discarded.
        """
        circuit_hashes: Dict[str, int] = {}  # map circuit_hash -> circuit_id
        circuits: Dict[int, Circuit] = {}  # map circuit_id -> circuit
        records = []
        for simulation_id, (index, simulation_path) in enumerate(simulation_paths.etl.iter()):
            simulation = Simulation(simulation_path)
            circuit_hash = cls._get_circuit_hash(simulation.circuit.config)
            # if circuit_hash is new, use simulation_id as circuit_id
            circuit_id = circuit_hashes.setdefault(circuit_hash, simulation_id)
            circuit = circuits.setdefault(circuit_id, simulation.circuit)
            sim_repr = f"{simulation_id=}, {circuit_id=}, {circuit_hash=}, {simulation_path=}"
            if not load_spikes or cls._is_simulation_complete(simulation):
                records.append(
                    {
                        SIMULATION_ID: simulation_id,
                        CIRCUIT_ID: circuit_id,
                        SIMULATION_PATH: simulation_path,
                        SIMULATION: simulation,
                        CIRCUIT: circuit,
                        **index._asdict(),
                    }
                )
                L.info("Extracting simulation: %s", sim_repr)
            else:
                L.warning("Ignoring simulation without spikes: %s", sim_repr)
        return pd.DataFrame(records).set_index(simulation_paths.index.names)

    @classmethod
    def from_config(cls, config: SimulationsConfig) -> "Simulations":
        """Load simulations from the given simulation campaign."""
        simulation_paths = config.to_pandas()
        df = cls._from_paths(simulation_paths, load_spikes=True)
        return cls(df)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "Simulations":
        """Load simulations from a dataframe containing valid simulation ids and circuit ids."""
        simulation_paths = df[SIMULATION_PATH]
        new_df = cls._from_paths(simulation_paths, load_spikes=False)
        check_columns = [SIMULATION_PATH, SIMULATION_ID, CIRCUIT_ID]
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
