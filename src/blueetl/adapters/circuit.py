"""Adapters for Circuit."""

from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from blueetl.adapters.base import BaseAdapter
from blueetl.adapters.interfaces.circuit import CircuitInterface, NodePopulationInterface
from blueetl.adapters.node_sets import NodeSetsAdapter


class CircuitAdapter(BaseAdapter[CircuitInterface]):
    """Circuit Adapter."""

    @classmethod
    def from_file(cls, filepath: Optional[Path]) -> "CircuitAdapter":
        """Load and return a new object from file."""
        # pylint: disable=import-outside-toplevel
        if not filepath or not filepath.exists():
            return cls(None)
        CircuitImpl: type[CircuitInterface]
        if filepath.suffix == ".json":
            from blueetl.adapters.impl.bluepysnap.circuit import Circuit, CircuitImpl
        else:
            from blueetl.adapters.impl.bluepy.circuit import Circuit, CircuitImpl
        impl = CircuitImpl(Circuit(str(filepath)))
        return cls(impl)

    def checksum(self) -> str:
        """Return a checksum of the relevant keys in the circuit configuration."""
        return self._ensure_impl.checksum()

    @property
    def nodes(self) -> Mapping[Optional[str], NodePopulationInterface]:
        """Return the nodes as a dict: population -> nodes."""
        return self._ensure_impl.nodes

    @property
    def node_sets_file(self) -> Optional[Path]:
        """Returns the NodeSets file used by the circuit."""
        return self._ensure_impl.node_sets_file

    @property
    def node_sets(self) -> NodeSetsAdapter:
        """Returns the NodeSets file used by the circuit."""
        return NodeSetsAdapter(self._ensure_impl.node_sets)
