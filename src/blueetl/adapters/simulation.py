"""Adapters for Simulation."""

from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from blueetl.adapters.base import BaseAdapter
from blueetl.adapters.circuit import CircuitAdapter
from blueetl.adapters.interfaces.simulation import (
    PopulationReportInterface,
    PopulationSpikesReportInterface,
    SimulationInterface,
)


class SimulationAdapter(BaseAdapter[SimulationInterface]):
    """Simulation Adapter."""

    @classmethod
    def from_file(cls, filepath: Optional[Path]) -> "SimulationAdapter":
        """Load and return a new object from file."""
        # pylint: disable=import-outside-toplevel
        if not filepath or not filepath.exists():
            return cls(None)
        SimulationImpl: type[SimulationInterface]
        if filepath.suffix == ".json":
            from blueetl.adapters.impl.bluepysnap.simulation import Simulation, SimulationImpl
        else:
            from blueetl.adapters.impl.bluepy.simulation import Simulation, SimulationImpl
        impl = SimulationImpl(Simulation(str(filepath)))
        return cls(impl)

    def is_complete(self) -> bool:
        """Return True if the simulation is complete, False otherwise."""
        return self.exists() and self._ensure_impl.is_complete()

    @property
    def circuit(self) -> CircuitAdapter:
        """Return the circuit used for the simulation."""
        return CircuitAdapter(self._ensure_impl.circuit)

    @property
    def spikes(self) -> Mapping[Optional[str], PopulationSpikesReportInterface]:
        """Return the spikes report as a dict: population -> report."""
        return self._ensure_impl.spikes

    @property
    def reports(self) -> Mapping[str, Mapping[Optional[str], PopulationReportInterface]]:
        """Return the reports as a dict: name -> population -> report."""
        return self._ensure_impl.reports
