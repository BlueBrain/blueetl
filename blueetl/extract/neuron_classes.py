"""NeuronClasses extractor."""
import logging
from typing import Dict, Optional

import pandas as pd

from blueetl.constants import CIRCUIT_ID, COUNT, NEURON_CLASS
from blueetl.extract.base import BaseExtractor
from blueetl.extract.neurons import Neurons

L = logging.getLogger(__name__)


class NeuronClasses(BaseExtractor):
    """NeuronClasses extractor class."""

    COLUMNS = [CIRCUIT_ID, NEURON_CLASS, COUNT]
    # allow additional columns containing the attributes of the neuron classes
    _allow_extra_columns = True

    @classmethod
    def from_neurons(
        cls,
        neurons: Neurons,
        target: Optional[str],
        neuron_classes: Dict[str, Dict],
        limit: Optional[int] = None,
    ) -> "NeuronClasses":
        """Load neuron classes information for each circuit.

        Args:
            neurons: Neurons extractor.
            target: target string, or None to not filter by target.
            neuron_classes: configuration dict of neuron classes to be extracted.
            limit: if specified, limit the number of extracted neurons.

        Returns:
            NeuronClasses: new instance.
        """
        results = []
        for index, count in neurons.count_by_neuron_class().etl.iter():
            # index: circuit_id, neuron_class
            neuron_class_conf = neuron_classes[index.neuron_class].copy()
            limit = neuron_class_conf.pop("$limit", limit)
            results.append(
                {
                    CIRCUIT_ID: index.circuit_id,
                    NEURON_CLASS: index.neuron_class,
                    COUNT: count,
                    "limit": limit,
                    "target": target,
                    **neuron_class_conf,
                }
            )
        df = pd.DataFrame(results)
        return cls(df)
