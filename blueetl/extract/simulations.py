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
    # allow additional columns containing the simulation conditions
    _allow_extra_columns = True

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
            # loading spikes can be slow, depending on their size
            _ = simulation.spikes
            return True
        except BluePyError:
            return False

    @classmethod
    def _from_paths(
        cls, simulation_paths: pd.DataFrame, ignore_spikes: bool = False
    ) -> pd.DataFrame:
        """Return a dataframe of simulations from a list of simulation paths.

        Args:
            simulation_paths: DataFrame containing the simulation paths.
                Any additional columns besides (simulation_path, simulation_id, circuit_id)
                are returned unchanged in the resulting DataFrame.
            ignore_spikes: If False, load the spikes to determine if the simulation is complete.
                If True, do not try to load the spikes and consider complete the simulations.

        Returns:
            DataFrame of simulations.
        """
        circuit_hashes: Dict[str, int] = {}  # map circuit_hash -> circuit_id
        circuits: Dict[int, Circuit] = {}  # map circuit_id -> circuit
        records = []
        for simulation_id, (_, rec) in enumerate(simulation_paths.etl.iterdict()):
            # use the cached simulation_id if available
            simulation_id = rec.get(SIMULATION_ID, simulation_id)
            # use the cached circuit_id if available, or fallback to simulation_id
            circuit_id = rec.get(CIRCUIT_ID, simulation_id)
            simulation_path = rec[SIMULATION_PATH]
            simulation = Simulation(simulation_path)
            circuit_hash = cls._get_circuit_hash(simulation.circuit.config)
            # if circuit_hash is not new, use the previous circuit_id
            circuit_id = circuit_hashes.setdefault(circuit_hash, circuit_id)
            circuit = circuits.setdefault(circuit_id, simulation.circuit)
            complete = ignore_spikes or cls._is_simulation_complete(simulation)
            sim_repr = f"{simulation_id=}, {circuit_id=}, {circuit_hash=}, {simulation_path=}"
            L.info("Extracting simulation: %s", sim_repr)
            records.append(
                {
                    **rec,  # must be first for lower precedence in case of collisions
                    SIMULATION_ID: simulation_id,
                    CIRCUIT_ID: circuit_id,
                    SIMULATION_PATH: simulation_path,
                    SIMULATION: simulation,
                    CIRCUIT: circuit,
                    "_complete": complete,
                }
            )
        if len(set(circuit_hashes.values())) != len(circuit_hashes):
            L.error("Multiple circuits have the same circuit_id, you may need to delete the cache.")
            raise RuntimeError("Inconsistent simulations")
        return pd.DataFrame(records)

    @classmethod
    def _filter_simulations_df(cls, df: pd.DataFrame, query: Optional[Dict]) -> pd.DataFrame:
        n_tot = len(df)
        # filter by _complete and drop the column because only for internal use
        df = df.etl.q({"_complete": True}).drop(columns="_complete")
        n1 = len(df)
        df = df.etl.q(query or {})
        n2 = len(df)
        L.info("%s simulations ignored because incomplete", n_tot - n1)
        L.info("%s simulations filtered out by the query: %s", n1 - n2, query)
        L.info("Extracted %s/%s simulations: %s", n2, n_tot, df[SIMULATION_ID].to_list())
        return df

    @classmethod
    def from_config(cls, config: SimulationsConfig, query: Optional[Dict] = None) -> "Simulations":
        """Load simulations from the given simulation campaign."""
        df = config.to_pandas()
        df = cls._from_paths(df, ignore_spikes=False)
        df = cls._filter_simulations_df(df, query)
        return cls(df)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, query: Optional[Dict] = None) -> "Simulations":
        """Load simulations from a dataframe containing valid simulation ids and circuit ids."""
        df = cls._from_paths(df, ignore_spikes=True)
        df = cls._filter_simulations_df(df, query)
        return cls(df)

    def to_pandas(self) -> pd.DataFrame:
        """Dump simulations to a dataframe that can be serialized and stored."""
        # skip columns SIMULATION and CIRCUIT because they contain custom runtime objects
        columns = [col for col in self.df.columns if col not in [SIMULATION, CIRCUIT]]
        return self.df[columns]
