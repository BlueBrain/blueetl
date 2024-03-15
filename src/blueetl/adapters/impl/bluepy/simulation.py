"""Bluepy simulation implementation."""

from collections import UserDict
from collections.abc import Mapping
from functools import cached_property
from typing import Optional, Union

import pandas as pd
from bluepy import Simulation
from bluepy.exceptions import BluePyError
from bluepy.impl.compartment_report import CompartmentReport, SomaReport
from bluepy.impl.spike_report import SpikeReport
from bluepy.simulation import PathHelpers

from blueetl.adapters.impl.bluepy.circuit import CircuitImpl
from blueetl.adapters.interfaces.circuit import CircuitInterface
from blueetl.adapters.interfaces.simulation import (
    PopulationReportInterface,
    PopulationSpikesReportInterface,
    SimulationInterface,
)


class PopulationSpikesReportImpl(PopulationSpikesReportInterface):
    """Bluepy spikes report implementation."""

    def __init__(self, spikes: SpikeReport) -> None:
        """Init the spikes report with the given spikes."""
        self._spikes = spikes

    def get(self, group=None, t_start=None, t_stop=None) -> pd.Series:
        """Return the spikes report for the specified group and interval.

        Only `group` as array of numeric ids is supported.
        """
        return self._spikes.get(gids=group, t_start=t_start, t_end=t_stop)


class PopulationReportImpl(PopulationReportInterface):
    """Bluepy report implementation."""

    def __init__(self, report: Union[SomaReport, CompartmentReport]) -> None:
        """Init the population report with the given report."""
        self._report = report

    def get(self, group=None, t_start=None, t_stop=None, t_step=None) -> pd.DataFrame:
        """Return the report for the specified group and interval.

        Only `group` as array of numeric ids is supported.
        """
        return self._report.get(gids=group, t_start=t_start, t_end=t_stop, t_step=t_step)


class ReportCollection(UserDict):
    """Collection of reports as: name -> population -> report."""

    def __init__(self, simulation: Simulation) -> None:
        """Init the report collection with the specified simulation."""
        super().__init__()
        self._simulation = simulation

    def __getitem__(self, name) -> Mapping[Optional[str], PopulationReportInterface]:
        """Return the report for the specified name, wrapped in a dict as: population -> report.

        The population name in the returned dict is always None, because undefined in the config.
        """
        if name not in self.data:
            self.data[name] = {None: PopulationReportImpl(self._simulation.report(name))}
        return self.data[name]


class SimulationImpl(SimulationInterface[Simulation]):
    """Bluepy simulation implementation."""

    def is_complete(self) -> bool:
        """Return True if the simulation is complete, False otherwise."""
        try:
            # check the existence of spikes without loading them, because it can be slow
            PathHelpers.spike_report_path(self._simulation.config)
            return True
        except BluePyError:
            return False

    @cached_property
    def circuit(self) -> CircuitInterface:
        """Return the circuit used for the simulation."""
        return CircuitImpl(self._simulation.circuit)

    @cached_property
    def spikes(self) -> Mapping[Optional[str], PopulationSpikesReportInterface]:
        """Return the spikes report as a dict: population -> report.

        The population name in the returned dict is always None, because undefined in the config.
        """
        return {None: PopulationSpikesReportImpl(self._simulation.spikes)}

    @cached_property
    def reports(self) -> Mapping[str, Mapping[Optional[str], PopulationReportInterface]]:
        """Return the reports as a dict: name -> population -> report.

        The population name in the returned dict is always None, because undefined in the config.
        """
        return ReportCollection(simulation=self._simulation)
