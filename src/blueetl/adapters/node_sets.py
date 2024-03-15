"""Adapters for NodeSets."""

from pathlib import Path
from typing import Optional

from blueetl.adapters.base import BaseAdapter
from blueetl.adapters.interfaces.node_sets import NodeSetsInterface


class NodeSetsAdapter(BaseAdapter[NodeSetsInterface]):
    """NodeSets Adapter."""

    @classmethod
    def from_file(cls, filepath: Optional[Path]) -> "NodeSetsAdapter":
        """Load and return a new object from file."""
        # pylint: disable=import-outside-toplevel
        from blueetl.adapters.impl.bluepysnap.node_sets import NodeSets, NodeSetsImpl

        if not filepath:
            impl = NodeSetsImpl(NodeSets.from_dict({}))
        else:
            impl = NodeSetsImpl(NodeSets.from_file(str(filepath)))
        return cls(impl)

    def update(self, other: "NodeSetsAdapter") -> None:
        """Update the node sets."""
        # pylint: disable=protected-access
        self._ensure_impl.update(other._ensure_impl)

    def __ior__(self, other: "NodeSetsAdapter") -> "NodeSetsAdapter":
        """Support ``A |= B``."""
        self.update(other)
        return self
