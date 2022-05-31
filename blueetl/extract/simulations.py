import hashlib
import json
import logging
from typing import Dict, Optional, Set

import pandas as pd
from bluepy import Circuit, Simulation
from bluepy.exceptions import BluePyError

from blueetl.config.simulations import SimulationsConfig
from blueetl.constants import CIRCUIT, CIRCUIT_ID, SIMULATION, SIMULATION_ID, SIMULATION_PATH
from blueetl.extract.base import BaseExtractor

L = logging.getLogger(__name__)


class Simulations(BaseExtractor):
    COLUMNS = [SIMULATION_PATH, SIMULATION_ID, CIRCUIT_ID, SIMULATION, CIRCUIT]

    @classmethod
    def _validate(cls, df):
        # allow additional arbitrary columns containing the simulation conditions
        cls._validate_columns(df, allow_extra=True)

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
    def _from_paths(
        cls,
        simulation_paths: pd.DataFrame,
        load_spikes: bool = True,
        simulation_ids: Optional[Set[int]] = None,
    ) -> pd.DataFrame:
        """Return a dataframe of simulations from a list of simulation paths.

        Args:
            simulation_paths: DataFrame containing the simulation paths.
                Any additional columns besides (simulation_path, simulation_id, circuit_id)
                are returned unchanged in the resulting DataFrame.
            load_spikes: if True, load the spikes to verify that they are available,
                and add the simulation to the resulting DataFrame only in that case.
                If False, don't load the spikes and always add the simulation.
            simulation_ids: if specified, load only the given simulations.

        Returns:
            DataFrame of simulations, using the same index of simulation_paths.
            Simulations without spikes are discarded.
        """
        circuit_hashes: Dict[str, int] = {}  # map circuit_hash -> circuit_id
        circuits: Dict[int, Circuit] = {}  # map circuit_id -> circuit
        records = []
        columns = simulation_paths.columns
        for simulation_id, (_, rec) in enumerate(simulation_paths.etl.iter()):
            rec = dict(zip(columns, rec))
            simulation_path = rec[SIMULATION_PATH]
            simulation = Simulation(simulation_path)
            circuit_hash = cls._get_circuit_hash(simulation.circuit.config)
            # if circuit_hash is new, use simulation_id as circuit_id
            circuit_id = circuit_hashes.setdefault(circuit_hash, simulation_id)
            circuit = circuits.setdefault(circuit_id, simulation.circuit)
            sim_repr = f"{simulation_id=}, {circuit_id=}, {circuit_hash=}, {simulation_path=}"
            if simulation_ids and simulation_id not in simulation_ids:
                L.warning("Ignoring simulation not selected by id: %s", sim_repr)
            elif load_spikes and not cls._is_simulation_complete(simulation):
                L.warning("Ignoring simulation without spikes: %s", sim_repr)
            else:
                L.info("Extracting simulation: %s", sim_repr)
                records.append(
                    {
                        **rec,  # must be first for lower precedence in case of collisions
                        SIMULATION_ID: simulation_id,
                        CIRCUIT_ID: circuit_id,
                        SIMULATION_PATH: simulation_path,
                        SIMULATION: simulation,
                        CIRCUIT: circuit,
                    }
                )
        return pd.DataFrame(records)

    @classmethod
    def from_config(
        cls, config: SimulationsConfig, simulation_ids: Optional[Set[int]] = None
    ) -> "Simulations":
        """Load simulations from the given simulation campaign."""
        df = config.to_pandas()
        new_df = cls._from_paths(df, load_spikes=True, simulation_ids=simulation_ids)
        return cls(new_df)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "Simulations":
        """Load simulations from a dataframe containing valid simulation ids and circuit ids."""
        new_df = cls._from_paths(df, load_spikes=False)
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
