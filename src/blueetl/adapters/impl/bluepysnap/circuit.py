"""Bluepysnap circuit implementation."""

from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from bluepysnap import Circuit

from blueetl.adapters.impl.bluepysnap.node_sets import NodeSetsImpl
from blueetl.adapters.interfaces.circuit import CircuitInterface, NodePopulationInterface
from blueetl.adapters.interfaces.node_sets import NodeSetsInterface
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

    @property
    def node_sets_file(self) -> Optional[Path]:
        """Returns the NodeSets file used by the circuit."""
        path = self._circuit.to_libsonata.node_sets_path
        return Path(path) if path else None

    @property
    def node_sets(self) -> NodeSetsInterface:
        """Returns the NodeSets used by the circuit."""
        return NodeSetsImpl(self._circuit.node_sets)
