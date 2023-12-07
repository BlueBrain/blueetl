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

    def _load_impl(self, config: str) -> Optional[SimulationInterface]:
        """Load and return the implementation object, or None if the config file doesn't exist."""
        # pylint: disable=import-outside-toplevel
        if not config or not Path(config).exists():
            return None
        SimulationImpl: type[SimulationInterface]
        if config.endswith(".json"):
            from blueetl.adapters.bluepysnap.simulation import Simulation, SimulationImpl
        else:
            from blueetl.adapters.bluepy.simulation import Simulation, SimulationImpl
        return SimulationImpl(Simulation(config))

    def is_complete(self) -> bool:
        """Return True if the simulation is complete, False otherwise."""
        return self.exists() and self._ensure_impl.is_complete()

    @property
    def circuit(self) -> CircuitAdapter:
        """Return the circuit used for the simulation."""
        return CircuitAdapter.from_impl(self._ensure_impl.circuit)

    @property
    def spikes(self) -> Mapping[Optional[str], PopulationSpikesReportInterface]:
        """Return the spikes report as a dict: population -> report."""
        return self._ensure_impl.spikes

    @property
    def reports(self) -> Mapping[str, Mapping[Optional[str], PopulationReportInterface]]:
        """Return the reports as a dict: name -> population -> report."""
        return self._ensure_impl.reports
