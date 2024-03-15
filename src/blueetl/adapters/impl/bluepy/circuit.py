"""Bluepy circuit implementation."""

from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from bluepy import Circuit

from blueetl.adapters.interfaces.circuit import CircuitInterface, NodePopulationInterface
from blueetl.adapters.interfaces.node_sets import NodeSetsInterface
from blueetl.utils import checksum_json


class CircuitImpl(CircuitInterface[Circuit]):
    """Bluepy circuit implementation."""

    def checksum(self) -> str:
        """Return a checksum of the relevant keys in the circuit configuration.

        Circuits are considered different in this context if any relevant key is different.
        """
        # the targets key isn't considered, since it may contain TargetFile defined in BlueConfig
        circuit_config_keys = [
            "cells",
            "morphologies",
            "morphology_type",
            "emodels",
            "mecombo_info",
            "connectome",
            "projections",
            "projections_metadata",
            "segment_index",
            "synapse_index",
            "atlas",
        ]
        return checksum_json({k: self._circuit.config.get(k) for k in circuit_config_keys})

    @property
    def nodes(self) -> Mapping[Optional[str], NodePopulationInterface]:
        """Return the nodes as a dict: population -> nodes.

        The population name in the returned dict is always None, because undefined in the config.
        """
        return {None: self._circuit.cells}

    @property
    def node_sets_file(self) -> Optional[Path]:
        """Returns the NodeSets file used by the circuit."""
        raise NotImplementedError

    @property
    def node_sets(self) -> NodeSetsInterface:
        """Returns the NodeSets used by the circuit."""
        raise NotImplementedError
