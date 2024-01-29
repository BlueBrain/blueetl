"""Bluepysnap circuit implementation."""

from collections.abc import Mapping
from typing import Optional

from bluepysnap import Circuit

from blueetl.adapters.interfaces.circuit import CircuitInterface, NodePopulationInterface
from blueetl.utils import checksum_json


class CircuitImpl(CircuitInterface[Circuit]):
    """Bluepysnap circuit implementation."""

    def checksum(self) -> str:
        """Return a hash of the relevant keys in the circuit configuration.

        Circuits are considered different in this context if any relevant key is different.
        """
        return checksum_json(self._circuit.config)

    @property
    def nodes(self) -> Mapping[Optional[str], NodePopulationInterface]:
        """Return the nodes as a dict: population -> nodes."""
        return self._circuit.nodes
