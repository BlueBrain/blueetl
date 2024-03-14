"""Bluepysnap simulation implementation."""

from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import Optional

from bluepysnap import Simulation

from blueetl.adapters.impl.bluepysnap.circuit import CircuitImpl
from blueetl.adapters.interfaces.circuit import CircuitInterface
from blueetl.adapters.interfaces.simulation import (
    PopulationReportInterface,
    PopulationSpikesReportInterface,
    SimulationInterface,
)


class SimulationImpl(SimulationInterface[Simulation]):
    """Bluepysnap simulation implementation."""

    def is_complete(self) -> bool:
        """Return True if the spikes can be loaded from the simulation, False otherwise.

        Used to ignore a simulation before the simulation campaign is complete.
        """
        config = self._simulation.spikes.config
        return Path(config.output_dir, config.spikes_file).exists()

    @cached_property
    def circuit(self) -> CircuitInterface:
        """Return the circuit used for the simulation."""
        return CircuitImpl(self._simulation.circuit)

    @property
    def spikes(self) -> Mapping[Optional[str], PopulationSpikesReportInterface]:
        """Return the spikes report as a dict: population -> report."""
        return self._simulation.spikes

    @property
    def reports(self) -> Mapping[str, Mapping[Optional[str], PopulationReportInterface]]:
        """Return the reports as a dict: name -> population -> report."""
        return self._simulation.reports
