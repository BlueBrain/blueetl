"""Simulations extractor."""
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from bluepy import Circuit, Simulation
from bluepy.exceptions import BluePyError
from bluepy.simulation import PathHelpers

from blueetl.config.simulations import SimulationsConfig
from blueetl.constants import CIRCUIT, CIRCUIT_ID, SIMULATION, SIMULATION_ID, SIMULATION_PATH
from blueetl.extract.base import BaseExtractor
from blueetl.extract.types import StrOrPath
from blueetl.utils import checksum_json

L = logging.getLogger(__name__)

# temporary column in the simulations dataframe
_STATUS = "_status"


def _get_circuit_hash(circuit_config: Dict[str, Any]) -> str:
    """Return a hash of the relevant keys in the circuit configuration.

    Circuits are considered different in this context if any relevant key is different.
    """
    # targets cannot be considered, since it may contain TargetFile defined in BlueConfig
    circuit_config_keys = [
        "cells",
        "morphologies",
        "morphology_type",
        "emodels",
        "mecombo_info",
        "connectome",
        # "targets",
        "projections",
        "projections_metadata",
        "segment_index",
        "synapse_index",
        "atlas",
    ]
    return checksum_json({k: circuit_config.get(k) for k in circuit_config_keys})


def _is_simulation_existing(simulation_path: StrOrPath) -> bool:
    """Return True if the simulation exists, False otherwise.

    Used to ignore a simulation because it was manually deleted / has gone missing.
    """
    return Path(simulation_path).exists()


def _is_simulation_complete(simulation: Simulation) -> bool:
    """Return True if the spikes can be loaded from the simulation, False otherwise.

    Used to ignore a simulation before the simulation campaign is complete.
    """
    try:
        # check the existence of spikes without loading them, because it can be slow
        PathHelpers.spike_report_path(simulation.config)
        return True
    except BluePyError:
        return False


class SimulationStatus(Enum):
    """Status of the loaded simulation."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    MISSING = "MISSING"


class Simulations(BaseExtractor):
    """Simulations extractor class."""

    COLUMNS = [SIMULATION_PATH, SIMULATION_ID, CIRCUIT_ID, SIMULATION, CIRCUIT]
    # allow additional columns containing the simulation conditions
    _allow_extra_columns = True

    @classmethod
    def _build_record(
        cls,
        simulation_id: int,
        rec: Dict[str, Any],
        circuit_hashes: Dict[str, int],
        circuits: Dict[int, Circuit],
    ) -> Dict[str, Any]:
        """Build a record to be added to the simulations dataframe."""
        # use the cached simulation_id if available
        simulation_id = rec.get(SIMULATION_ID, simulation_id)
        # use the cached circuit_id if available, or fallback to simulation_id
        circuit_id = rec.get(CIRCUIT_ID, simulation_id)
        simulation_path = rec[SIMULATION_PATH]
        simulation = circuit = circuit_hash = None
        status = SimulationStatus.MISSING
        if _is_simulation_existing(simulation_path):
            status = SimulationStatus.INCOMPLETE
            simulation = Simulation(simulation_path)
            circuit_hash = _get_circuit_hash(simulation.circuit.config)
            # if circuit_hash is not new, use the previous circuit_id
            circuit_id = circuit_hashes.setdefault(circuit_hash, circuit_id)
            circuit = circuits.setdefault(circuit_id, simulation.circuit)
            if _is_simulation_complete(simulation):
                status = SimulationStatus.COMPLETE
        sim_repr = f"{simulation_id=}, {circuit_id=}, {circuit_hash=}, {simulation_path=}"
        L.info("Processing simulation: %s", sim_repr)
        return {
            SIMULATION_ID: simulation_id,
            CIRCUIT_ID: circuit_id,
            SIMULATION_PATH: simulation_path,
            SIMULATION: simulation,
            CIRCUIT: circuit,
            _STATUS: status,
        }

    @classmethod
    def _from_paths(cls, simulation_paths: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe of simulations from a list of simulation paths.

        Args:
            simulation_paths: DataFrame containing the simulation paths.
                Any additional columns besides (simulation_path, simulation_id, circuit_id)
                are returned unchanged in the resulting DataFrame.

        Returns:
            DataFrame of simulations.
        """
        circuit_hashes: Dict[str, int] = {}  # map circuit_hash -> circuit_id
        circuits: Dict[int, Circuit] = {}  # map circuit_id -> circuit
        records = []
        for simulation_id, (_, rec) in enumerate(simulation_paths.etl.iterdict()):
            record = cls._build_record(
                simulation_id=simulation_id,
                rec=rec,
                circuit_hashes=circuit_hashes,
                circuits=circuits,
            )
            # rec must be first for lower precedence in case of collisions
            records.append({**rec, **record})
        if len(set(circuit_hashes.values())) != len(circuit_hashes):
            L.error("Multiple circuits have the same circuit_id, you may need to delete the cache.")
            raise RuntimeError("Inconsistent simulations")
        return pd.DataFrame(records)

    @classmethod
    def _filter_simulations_df(
        cls, df: pd.DataFrame, query: Optional[Dict], cached: bool = False
    ) -> pd.DataFrame:
        """Remove the missing and incomplete simulations and filter by the given query.

        Args:
            df: DataFrame to filter.
            query: filtering query.
            cached: if True, raise an error in case any simulation is missing or incomplete.

        Returns:
            the filtered DataFrame.
        """
        len_total = len(df)
        len_missing = len(df.etl.q({_STATUS: SimulationStatus.MISSING}))
        len_incomplete = len(df.etl.q({_STATUS: SimulationStatus.INCOMPLETE}))
        # filter by the internal ``_status`` column, and drop the column
        df = df.etl.q({_STATUS: SimulationStatus.COMPLETE}).drop(columns=_STATUS)
        len_complete = len(df)
        assert len_missing + len_incomplete + len_complete == len_total
        # filter by the custom query if provided
        df = df.etl.q(query or {})
        # reset the index to ensure that it doesn't contain gaps,
        # while preserving the simulation_id in a dedicated column
        df = df.reset_index(drop=True)
        len_final = len(df)
        L.info("Simulations ignored because missing: %s", len_missing)
        L.info("Simulations ignored because incomplete: %s", len_incomplete)
        L.info("Simulations filtered out: %s, with query: %s", len_complete - len_final, query)
        L.info(
            "Simulations extracted: %s/%s, ids: %s",
            len_final,
            len_total,
            df[SIMULATION_ID].to_list(),
        )
        if cached and (len_missing or len_incomplete):
            L.error(
                "Some cached simulations are missing or incomplete, "
                "you may need to delete the cache."
            )
            raise RuntimeError("Inconsistent cached simulations")
        return df

    @classmethod
    def from_config(cls, config: SimulationsConfig, query: Optional[Dict] = None) -> "Simulations":
        """Extract simulations from the given simulation campaign."""
        df = config.to_pandas()
        df = cls._from_paths(df)
        df = cls._filter_simulations_df(df, query)
        return cls(df)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, query: Optional[Dict] = None) -> "Simulations":
        """Extract simulations from a dataframe containing valid simulation ids and circuit ids."""
        df = cls._from_paths(df)
        df = cls._filter_simulations_df(df, query, cached=True)
        return cls(df)

    def to_pandas(self) -> pd.DataFrame:
        """Dump simulations to a dataframe that can be serialized and stored."""
        # skip columns SIMULATION and CIRCUIT because they contain custom runtime objects
        columns = [col for col in self.df.columns if col not in [SIMULATION, CIRCUIT]]
        return self.df[columns]
