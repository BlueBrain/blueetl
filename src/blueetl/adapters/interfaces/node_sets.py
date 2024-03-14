"""Interfaces for NodeSets."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

NodeSetsT = TypeVar("NodeSetsT")


class NodeSetsInterface(Generic[NodeSetsT], ABC):
    """NodeSets Interface."""

    def __init__(self, node_sets: NodeSetsT) -> None:
        """Init the NodeSets interface with the specified NodeSetsT."""
        self._node_sets = node_sets

    @property
    def instance(self) -> NodeSetsT:
        """Return the wrapped instance."""
        return self._node_sets

    @abstractmethod
    def update(self, other: "NodeSetsInterface") -> None:
        """Update the wrapped node sets."""
