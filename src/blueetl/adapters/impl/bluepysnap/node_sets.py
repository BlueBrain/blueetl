"""Bluepysnap NodeSets implementation."""

from bluepysnap.node_sets import NodeSets

from blueetl.adapters.interfaces.node_sets import NodeSetsInterface


class NodeSetsImpl(NodeSetsInterface[NodeSets]):
    """Bluepysnap NodeSets implementation."""

    def update(self, other: NodeSetsInterface) -> None:
        """Update the wrapped node sets."""
        self._node_sets.update(other.instance)
