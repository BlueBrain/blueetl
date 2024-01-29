"""Adapters for Circuit."""

from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from blueetl.adapters.base import BaseAdapter
from blueetl.adapters.interfaces.circuit import CircuitInterface, NodePopulationInterface


class CircuitAdapter(BaseAdapter[CircuitInterface]):
    """Circuit Adapter."""

    def _load_impl(self, config: str) -> Optional[CircuitInterface]:
        """Load and return the implementation object, or None if the config file doesn't exist."""
        # pylint: disable=import-outside-toplevel
        if not config or not Path(config).exists():
            return None
        CircuitImpl: type[CircuitInterface]
        if config.endswith(".json"):
            from blueetl.adapters.bluepysnap.circuit import Circuit, CircuitImpl
        else:
            from blueetl.adapters.bluepy.circuit import Circuit, CircuitImpl
        return CircuitImpl(Circuit(config))

    def checksum(self) -> str:
        """Return a checksum of the relevant keys in the circuit configuration."""
        return self._ensure_impl.checksum()

    @property
    def nodes(self) -> Mapping[Optional[str], NodePopulationInterface]:
        """Return the nodes as a dict: population -> nodes."""
        return self._ensure_impl.nodes
