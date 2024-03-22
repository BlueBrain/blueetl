"""Simulations extractor."""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd

from blueetl.adapters.circuit import CircuitAdapter as Circuit
from blueetl.adapters.simulation import SimulationAdapter as Simulation
from blueetl.campaign.config import SimulationCampaign
from blueetl.constants import CIRCUIT, CIRCUIT_ID, SIMULATION, SIMULATION_ID, SIMULATION_PATH
from blueetl.extract.base import BaseExtractor

L = logging.getLogger(__name__)

# temporary column in the simulations dataframe
_STATUS = "_status"


class SimulationStatus(Enum):
    """Status of the loaded simulation."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    MISSING = "MISSING"


class InconsistentSimulations(Exception):
    """Error raised when the extracted simulations have some inconsistencies."""


class Simulations(BaseExtractor):
    """Simulations extractor class."""

    COLUMNS = [SIMULATION_PATH, SIMULATION_ID, CIRCUIT_ID, SIMULATION, CIRCUIT]
    # allow additional columns containing the simulation conditions
    _allow_extra_columns = True

    @classmethod
    def _build_record(
        cls,
        simulation_id: int,
        rec: dict[str, Any],
        circuit_hashes: dict[str, int],
        circuits: dict[int, Circuit],
    ) -> dict[str, Any]:
        """Build a record to be added to the simulations dataframe."""
        # use the cached simulation_id if available
        simulation_id = rec.get(SIMULATION_ID, simulation_id)
        # use the cached circuit_id if available, or fallback to simulation_id
        circuit_id = rec.get(CIRCUIT_ID, simulation_id)
        simulation_path = rec[SIMULATION_PATH]
        # use the same Simulation and Circuit objects if available
        simulation = cast(Optional[Simulation], rec.get(SIMULATION))
        circuit = cast(Optional[Circuit], rec.get(CIRCUIT))
        assert (
            simulation and circuit or not simulation and not circuit
        ), "Simulation and Circuit must be both initialized, or both not initialized"
        circuit_hash = None
        status = SimulationStatus.MISSING
        simulation = simulation or Simulation.from_file(Path(simulation_path))
        if simulation.exists():
            # consider the simulation only if it wasn't manually deleted
            status = SimulationStatus.INCOMPLETE
            circuit_hash = simulation.circuit.checksum()
            # if circuit_hash is not new, use the previous circuit_id
            circuit_id = circuit_hashes.setdefault(circuit_hash, circuit_id)
            circuit = circuit or circuits.setdefault(circuit_id, simulation.circuit)
            # double-check the circuit config hash in case it was cached
            if circuit_hash != circuit.checksum():
                L.error("Inconsistent circuit hash and id, you may need to delete the cache")
                raise InconsistentSimulations("Inconsistent hash and id")
            if simulation.is_complete():
                status = SimulationStatus.COMPLETE
        sim_repr = f"{simulation_id=}, {circuit_id=}, {circuit_hash=}, {simulation_path=}"
        L.debug("Processing simulation: %s", sim_repr)
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
        circuit_hashes: dict[str, int] = {}  # map circuit_hash -> circuit_id
        circuits: dict[int, Circuit] = {}  # map circuit_id -> circuit
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
        assert len(set(circuit_hashes.values())) == len(circuit_hashes), "Inconsistent circuit ids"
        return pd.DataFrame(records)

    @classmethod
    def _filter_simulations_df(
        cls, df: pd.DataFrame, query: Optional[dict], cached: bool = False
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
            raise InconsistentSimulations("Inconsistent cache")
        return df

    @classmethod
    def from_config(cls, config: SimulationCampaign, query: Optional[dict] = None) -> "Simulations":
        """Extract simulations from the given simulation campaign."""
        df = config.get()
        return cls.from_pandas(df=df, query=query, cached=False)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        query: Optional[dict] = None,
        cached: bool = True,
    ) -> "Simulations":
        """Extract simulations from a dataframe containing valid simulation ids and circuit ids."""
        original_len = len(df)
        df = cls._from_paths(df)
        df = cls._filter_simulations_df(df, query, cached=cached)
        filtered = len(df) != original_len
        return cls(df, cached=cached, filtered=filtered)

    def to_pandas(self) -> pd.DataFrame:
        """Dump simulations to a dataframe that can be serialized and stored."""
        # skip columns SIMULATION and CIRCUIT because they contain custom runtime objects
        columns = [col for col in self.df.columns if col not in [SIMULATION, CIRCUIT]]
        return self.df[columns]
